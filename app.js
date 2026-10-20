/*
 * Vector Perception Demo — app.js
 *
 * Runs 100% client-side. Two modes:
 *   webcam  — real video + real TensorFlow.js inference (COCO-SSD + MoveNet)
 *   sim     — animated synthetic warehouse scene with simulated overlays
 *
 * Every model call is guarded: if the CDN is unreachable or the camera is
 * blocked, the app falls back to the simulated scene and keeps animating.
 */
(function () {
  'use strict';

  // ---------------------------------------------------------------- DOM ----
  var feedCanvas = document.getElementById('feedCanvas');
  var depthCanvas = document.getElementById('depthCanvas');
  var feedCtx = feedCanvas.getContext('2d');
  var depthCtx = depthCanvas.getContext('2d');
  var video = document.getElementById('camVideo');
  var notice = document.getElementById('notice');
  var simBadge = document.getElementById('simBadge');
  var modeText = document.getElementById('modeText');
  var btnWebcam = document.getElementById('btnWebcam');
  var btnSample = document.getElementById('btnSample');
  var tFps = document.getElementById('tFps');
  var tLatency = document.getElementById('tLatency');
  var tDetections = document.getElementById('tDetections');
  var tTrackIds = document.getElementById('tTrackIds');

  var FW = feedCanvas.width;   // 640
  var FH = feedCanvas.height;  // 480
  var DW = depthCanvas.width;  // 320
  var DH = depthCanvas.height; // 240

  // --------------------------------------------------------------- state ---
  var mode = 'sim';              // 'webcam' | 'sim'
  var cameraOk = false;
  var cocoModel = null;
  var poseDetector = null;
  var tracks = [];               // IoU tracker state
  var nextTrackId = 1;
  var activeDetections = [];     // last detection list, in canvas coords
  var activePose = null;         // last MoveNet pose
  var lastInferAt = 0;
  var lastLatencyMs = null;
  var inferRunning = false;

  // FPS (exponential moving average of rAF delta)
  var fpsEma = null;
  var prevFrameAt = null;

  // ------------------------------------------------------ status badges ----
  function setBadge(id, state, label) {
    var el = document.getElementById(id);
    if (!el) return;
    el.classList.remove('is-loading', 'is-ready', 'is-failed');
    if (state) el.classList.add('is-' + state);
    var b = el.querySelector('b');
    if (b) b.textContent = label;
  }

  function setModeBadge(live) {
    modeText.textContent = live ? 'LIVE' : 'SIMULATED';
  }

  function showNotice(text) {
    if (!text) { notice.hidden = true; notice.textContent = ''; return; }
    notice.hidden = false;
    notice.textContent = text;
  }

  // ---------------------------------------------------------------- camera -
  async function tryCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      showNotice('Camera API unavailable in this browser — showing simulated feed.');
      return false;
    }
    try {
      var stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
        audio: false
      });
      video.srcObject = stream;
      await video.play().catch(function () {});
      cameraOk = true;
      return true;
    } catch (err) {
      cameraOk = false;
      showNotice('Camera unavailable (' + (err && err.name ? err.name : 'error') + ') — showing simulated feed.');
      return false;
    }
  }

  function setMode(next) {
    mode = next;
    var live = (mode === 'webcam' && cameraOk);
    simBadge.hidden = live;
    setModeBadge(live);
    btnWebcam.classList.toggle('btn-secondary', live);
    btnSample.classList.toggle('btn-secondary', !live);
  }

  btnWebcam.addEventListener('click', async function () {
    showNotice('Requesting camera…');
    var ok = await tryCamera();
    if (ok) {
      showNotice('');
      setMode('webcam');
    } else {
      setMode('sim');
    }
  });

  btnSample.addEventListener('click', function () {
    showNotice('');
    setMode('sim');
  });

  // ---------------------------------------------------------------- models -
  // Each model loads independently; a failure only flips its badge to "failed".
  async function loadModels() {
    if (typeof tf === 'undefined') {
      setBadge('badgeTfjs', 'failed', 'failed');
      setBadge('badgeCoco', 'failed', 'failed');
      setBadge('badgePose', 'failed', 'failed');
      return;
    }
    try {
      if (tf.setBackend) await tf.setBackend('webgl').catch(function () {});
      await tf.ready();
      setBadge('badgeTfjs', 'ready', 'ready');
    } catch (e) {
      setBadge('badgeTfjs', 'failed', 'failed');
      setBadge('badgeCoco', 'failed', 'failed');
      setBadge('badgePose', 'failed', 'failed');
      return;
    }

    if (typeof cocoSsd !== 'undefined') {
      try {
        cocoModel = await cocoSsd.load();
        setBadge('badgeCoco', 'ready', 'ready');
      } catch (e) {
        setBadge('badgeCoco', 'failed', 'failed');
      }
    } else {
      setBadge('badgeCoco', 'failed', 'failed');
    }

    if (typeof poseDetection !== 'undefined') {
      try {
        poseDetector = await poseDetection.createDetector(
          poseDetection.SupportedModels.MoveNet,
          { modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING }
        );
        setBadge('badgePose', 'ready', 'ready');
      } catch (e) {
        setBadge('badgePose', 'failed', 'failed');
      }
    } else {
      setBadge('badgePose', 'failed', 'failed');
    }
  }

  // ------------------------------------------------------------ IoU track -
  function iou(a, b) {
    var ax2 = a.x + a.w, ay2 = a.y + a.h;
    var bx2 = b.x + b.w, by2 = b.y + b.h;
    var ix1 = Math.max(a.x, b.x), iy1 = Math.max(a.y, b.y);
    var ix2 = Math.min(ax2, bx2), iy2 = Math.min(ay2, by2);
    var iw = Math.max(0, ix2 - ix1), ih = Math.max(0, iy2 - iy1);
    var inter = iw * ih;
    var union = a.w * a.h + b.w * b.h - inter;
    return union <= 0 ? 0 : inter / union;
  }

  // Greedy IoU matching. Boxes: [{x,y,w,h,label,score}].
  // Returns the same boxes with stable integer ids attached.
  function updateTracks(boxes) {
    var matched = new Array(boxes.length).fill(-1);

    boxes.forEach(function (box, bi) {
      var best = -1, bestIou = 0.35;
      tracks.forEach(function (t, ti) {
        if (t.matchedThisFrame) return;
        var v = iou(box, t.bbox);
        if (v > bestIou) { bestIou = v; best = ti; }
      });
      if (best >= 0) {
        tracks[best].bbox = { x: box.x, y: box.y, w: box.w, h: box.h };
        tracks[best].missed = 0;
        tracks[best].matchedThisFrame = true;
        matched[bi] = best;
      }
    });

    boxes.forEach(function (box, bi) {
      if (matched[bi] >= 0) {
        box.id = tracks[matched[bi]].id;
      } else {
        var t = { id: nextTrackId++, bbox: { x: box.x, y: box.y, w: box.w, h: box.h }, missed: 0, matchedThisFrame: true };
        tracks.push(t);
        box.id = t.id;
      }
    });

    // Age out unmatched tracks; drop after 5 missed frames.
    tracks.forEach(function (t) {
      if (!t.matchedThisFrame) t.missed += 1;
      t.matchedThisFrame = false;
    });
    tracks = tracks.filter(function (t) { return t.missed < 5; });

    return boxes;
  }

  // ------------------------------------------------------------- inference -
  var INFER_INTERVAL = 200; // ms

  async function runInference() {
    if (inferRunning) return;
    if (!cocoModel && !poseDetector) return;
    if (video.readyState < 2 || video.videoWidth === 0) return;
    if (performance.now() - lastInferAt < INFER_INTERVAL) return;

    inferRunning = true;
    lastInferAt = performance.now();
    var t0 = performance.now();
    try {
      var dets = [];
      if (cocoModel) {
        var preds = await cocoModel.detect(video);
        var sx = FW / video.videoWidth, sy = FH / video.videoHeight;
        dets = preds.map(function (p) {
          return {
            x: p.bbox[0] * sx, y: p.bbox[1] * sy,
            w: p.bbox[2] * sx, h: p.bbox[3] * sy,
            label: p.class, score: p.score
          };
        });
      }
      activeDetections = updateTracks(dets);

      if (poseDetector) {
        var poses = await poseDetector.estimatePoses(video);
        activePose = (poses && poses.length > 0) ? poses[0] : null;
        if (activePose) {
          var sx2 = FW / video.videoWidth, sy2 = FH / video.videoHeight;
          activePose.keypoints = activePose.keypoints.map(function (k) {
            return { x: k.x * sx2, y: k.y * sy2, score: k.score, name: k.name };
          });
        }
      }
      lastLatencyMs = performance.now() - t0;
    } catch (e) {
      // One bad frame must never kill the loop.
    }
    inferRunning = false;
  }

  // ------------------------------------------------------------- drawing ---
  var LABEL_FONT = '12px Inter, system-ui, sans-serif';

  function drawBox(ctx, b, dashed, tag) {
    ctx.save();
    ctx.strokeStyle = '#00f7ff';
    ctx.lineWidth = 2;
    ctx.setLineDash(dashed ? [6, 4] : []);
    ctx.strokeRect(b.x, b.y, b.w, b.h);
    ctx.setLineDash([]);
    var text = b.label + ' ' + Math.round(b.score * 100) + '%' + tag +
               (b.id ? '  ·  ID ' + b.id : '');
    ctx.font = LABEL_FONT;
    var tw = ctx.measureText(text).width + 12;
    ctx.fillStyle = '#00f7ff';
    ctx.fillRect(b.x, Math.max(0, b.y - 20), tw, 20);
    ctx.fillStyle = '#030712';
    ctx.fillText(text, b.x + 6, Math.max(0, b.y - 6));
    ctx.restore();
  }

  // MoveNet keypoint adjacency pairs.
  var SKELETON_EDGES = [
    ['left_shoulder', 'right_shoulder'], ['left_shoulder', 'left_elbow'],
    ['left_elbow', 'left_wrist'], ['right_shoulder', 'right_elbow'],
    ['right_elbow', 'right_wrist'], ['left_shoulder', 'left_hip'],
    ['right_shoulder', 'right_hip'], ['left_hip', 'right_hip'],
    ['left_hip', 'left_knee'], ['left_knee', 'left_ankle'],
    ['right_hip', 'right_knee'], ['right_knee', 'right_ankle'],
    ['nose', 'left_eye'], ['nose', 'right_eye'],
    ['left_eye', 'left_ear'], ['right_eye', 'right_ear']
  ];

  function drawSkeleton(ctx, keypoints, simulated) {
    var byName = {};
    keypoints.forEach(function (k) { byName[k.name] = k; });
    ctx.save();
    ctx.strokeStyle = '#06b6d4';
    ctx.lineWidth = 2;
    if (simulated) ctx.setLineDash([4, 3]);
    SKELETON_EDGES.forEach(function (edge) {
      var a = byName[edge[0]], b = byName[edge[1]];
      if (a && b && a.score > 0.3 && b.score > 0.3) {
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
      }
    });
    ctx.fillStyle = '#00f7ff';
    keypoints.forEach(function (k) {
      if (k.score > 0.3) {
        ctx.beginPath(); ctx.arc(k.x, k.y, 3.5, 0, Math.PI * 2); ctx.fill();
      }
    });
    ctx.restore();
  }

  // --------------------------------------------------------- sim scene ----
  // Synthetic warehouse: dark floor grid, drifting carton/pallet boxes,
  // one moving "person" capsule with a stick-figure skeleton.
  var simObjects = [
    { label: 'carton', x: 80, y: 300, w: 110, h: 90, vx: 0.55, score: 0.92, id: 101 },
    { label: 'pallet', x: 420, y: 330, w: 150, h: 70, vx: -0.4, score: 0.88, id: 102 },
    { label: 'carton', x: 300, y: 150, w: 80, h: 64, vx: 0.7, score: 0.9, id: 103 }
  ];
  var simPerson = { x: 200, y: 240, vx: 0.9, dir: 1, score: 0.95, id: 104 };

  function drawSimScene(t) {
    // Background
    feedCtx.fillStyle = '#030712';
    feedCtx.fillRect(0, 0, FW, FH);

    // Floor grid
    feedCtx.strokeStyle = '#1e293b';
    feedCtx.lineWidth = 1;
    var horizon = FH * 0.55;
    for (var gx = 0; gx <= FW; gx += 40) {
      feedCtx.beginPath(); feedCtx.moveTo(gx, horizon); feedCtx.lineTo(gx, FH); feedCtx.stroke();
    }
    for (var gy = horizon; gy <= FH; gy += 24) {
      feedCtx.beginPath(); feedCtx.moveTo(0, gy); feedCtx.lineTo(FW, gy); feedCtx.stroke();
    }
    // Back wall
    feedCtx.fillStyle = '#0f172a';
    feedCtx.fillRect(0, 0, FW, horizon);

    // Drift the boxes, wrap at edges
    simObjects.forEach(function (o) {
      o.x += o.vx;
      if (o.x > FW + 20) o.x = -o.w - 20;
      if (o.x < -o.w - 20) o.x = FW + 20;
      // box body
      feedCtx.fillStyle = '#1e293b';
      feedCtx.fillRect(o.x, o.y, o.w, o.h);
      feedCtx.strokeStyle = '#334155';
      feedCtx.strokeRect(o.x, o.y, o.w, o.h);
    });

    // Person proxy: capsule with bobbing motion
    var px = simPerson.x + Math.sin(t / 900) * 130;
    var bob = Math.abs(Math.sin(t / 220)) * 6;
    var py = simPerson.y - bob;
    feedCtx.fillStyle = '#12303a';
    feedCtx.strokeStyle = '#0fa3b3';
    feedCtx.lineWidth = 2;
    feedCtx.beginPath();
    if (feedCtx.roundRect) feedCtx.roundRect(px - 22, py - 70, 44, 150, 22);
    else feedCtx.rect(px - 22, py - 70, 44, 150);
    feedCtx.fill(); feedCtx.stroke();

    // Simulated overlays (dashed boxes, explicit "(simulated)" labels)
    var simDets = [];
    simObjects.forEach(function (o) {
      var b = { x: o.x, y: o.y, w: o.w, h: o.h, label: o.label, score: o.score, id: o.id };
      drawBox(feedCtx, b, true, ' (simulated)');
      simDets.push(b);
    });
    var personBox = { x: px - 26, y: py - 78, w: 52, h: 164, label: 'person', score: simPerson.score, id: simPerson.id };
    drawBox(feedCtx, personBox, true, ' (simulated)');
    simDets.push(personBox);

    // Stick figure on the person proxy
    var headY = py - 92;
    var kp = [
      { name: 'nose', x: px, y: headY - 8, score: 0.99 },
      { name: 'left_shoulder', x: px - 12, y: headY + 8, score: 0.99 },
      { name: 'right_shoulder', x: px + 12, y: headY + 8, score: 0.99 },
      { name: 'left_elbow', x: px - 20, y: headY + 28, score: 0.99 },
      { name: 'right_elbow', x: px + 20, y: headY + 28, score: 0.99 },
      { name: 'left_wrist', x: px - 18, y: headY + 48, score: 0.99 },
      { name: 'right_wrist', x: px + 18, y: headY + 48, score: 0.99 },
      { name: 'left_hip', x: px - 8, y: headY + 56, score: 0.99 },
      { name: 'right_hip', x: px + 8, y: headY + 56, score: 0.99 },
      { name: 'left_knee', x: px - 8, y: headY + 90, score: 0.99 },
      { name: 'right_knee', x: px + 8, y: headY + 90, score: 0.99 },
      { name: 'left_ankle', x: px - 8, y: headY + 118, score: 0.99 },
      { name: 'right_ankle', x: px + 8, y: headY + 118, score: 0.99 }
    ];
    drawSkeleton(feedCtx, kp, true);

    activeDetections = simDets;
    activePose = null;
  }

  // ------------------------------------------------------------- depth ------
  function drawDepth() {
    // Dark blue base
    depthCtx.fillStyle = '#071022';
    depthCtx.fillRect(0, 0, DW, DH);

    // Faint grid
    depthCtx.strokeStyle = 'rgba(51, 65, 85, 0.5)';
    depthCtx.lineWidth = 1;
    for (var x = 0; x <= DW; x += 32) {
      depthCtx.beginPath(); depthCtx.moveTo(x, 0); depthCtx.lineTo(x, DH); depthCtx.stroke();
    }
    for (var y = 0; y <= DH; y += 24) {
      depthCtx.beginPath(); depthCtx.moveTo(0, y); depthCtx.lineTo(DW, y); depthCtx.stroke();
    }

    // One soft radial blob per active detection.
    // Bigger bbox (on the 640x480 feed) = nearer = brighter, larger blob.
    var sx = DW / FW, sy = DH / FH;
    activeDetections.forEach(function (b) {
      var area = Math.max(0.01, (b.w * b.h) / (FW * FH)); // 0..~1
      var brightness = Math.min(1, 0.25 + area * 6);
      var cx = (b.x + b.w / 2) * sx, cy = (b.y + b.h / 2) * sy;
      var r = Math.max(18, Math.sqrt(b.w * b.h) * sx * 0.9);
      var g = depthCtx.createRadialGradient(cx, cy, 0, cx, cy, r);
      g.addColorStop(0, 'rgba(0, 247, 255, ' + brightness.toFixed(2) + ')');
      g.addColorStop(1, 'rgba(0, 247, 255, 0)');
      depthCtx.fillStyle = g;
      depthCtx.beginPath(); depthCtx.arc(cx, cy, r, 0, Math.PI * 2); depthCtx.fill();
    });
  }

  // ---------------------------------------------------------- telemetry ----
  function updateTelemetry() {
    tFps.textContent = fpsEma === null ? '—' : Math.round(fpsEma).toString();
    tLatency.textContent = lastLatencyMs === null ? '—' : Math.round(lastLatencyMs) + ' ms';
    tDetections.textContent = String(activeDetections.length);
    var ids = activeDetections.map(function (b) { return 'ID ' + b.id; });
    tTrackIds.textContent = ids.length ? ids.join(', ') : '—';
  }

  // --------------------------------------------------------------- loop ----
  async function frame(t) {
    // FPS EMA
    if (prevFrameAt !== null && t > prevFrameAt) {
      var fps = 1000 / (t - prevFrameAt);
      fpsEma = (fpsEma === null) ? fps : fpsEma * 0.9 + fps * 0.1;
    }
    prevFrameAt = t;

    if (mode === 'webcam' && cameraOk && video.readyState >= 2 && video.videoWidth > 0) {
      feedCtx.drawImage(video, 0, 0, FW, FH);
      await runInference();
      activeDetections.forEach(function (b) { drawBox(feedCtx, b, false, ''); });
      if (activePose && activePose.keypoints) drawSkeleton(feedCtx, activePose.keypoints, false);
    } else {
      drawSimScene(t);
    }

    drawDepth();
    updateTelemetry();
    requestAnimationFrame(frame);
  }

  // ---------------------------------------------------------------- boot ---
  setMode('sim');
  requestAnimationFrame(frame);
  loadModels();
  tryCamera().then(function (ok) {
    if (ok) { showNotice(''); setMode('webcam'); }
    else setMode('sim');
  });
})();
