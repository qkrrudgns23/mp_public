(function() {
  'use strict';

  var CFG = window.__SIM_VIEWER_CONFIG__ || {};
  var SIM = window.__SIM_PLAYBACK_DATA__ || {};
  var layout = CFG.layout || SIM.layout || {};
  var positions = SIM.positions || {};
  var GRID_COLS = CFG.gridCols || 200;
  var GRID_ROWS = CFG.gridRows || 200;
  var CELL_SIZE = CFG.cellSize || 20;

  var canvas = document.getElementById('viewer-canvas');
  var ctx = canvas ? canvas.getContext('2d') : null;
  var slider = document.getElementById('viewer-slider');
  var timeLabel = document.getElementById('viewer-time-label');
  var btnPlay = document.getElementById('btnPlay');
  var btnPause = document.getElementById('btnPause');
  var speedSel = document.getElementById('playSpeed');
  var infoDiv = document.getElementById('viewer-info');

  var playing = false;
  var playSpeed = 1;
  var currentTimeIdx = 0;
  var animFrameId = null;
  var lastFrameTs = 0;

  // --- build time axis from positions data ---
  var allTimes = [];
  var timeSet = {};
  Object.keys(positions).forEach(function(fid) {
    var pts = positions[fid];
    if (!Array.isArray(pts)) return;
    pts.forEach(function(p) {
      var t = p.t;
      if (t != null && !timeSet[t]) {
        timeSet[t] = true;
        allTimes.push(t);
      }
    });
  });
  allTimes.sort(function(a, b) { return a - b; });

  var totalFrames = allTimes.length;
  if (slider) slider.max = Math.max(totalFrames - 1, 0);

  // build per-flight lookup: fid -> sorted array of {t, col, row}
  var flightTimelines = {};
  Object.keys(positions).forEach(function(fid) {
    var pts = positions[fid];
    if (!Array.isArray(pts)) { flightTimelines[fid] = []; return; }
    flightTimelines[fid] = pts.slice().sort(function(a, b) { return a.t - b.t; });
  });

  // --- camera state (pan + zoom) ---
  var camX = 0;
  var camY = 0;
  var camScale = 1;

  function fitView() {
    if (!canvas) return;
    var w = canvas.width;
    var h = canvas.height;
    var worldW = GRID_COLS * CELL_SIZE;
    var worldH = GRID_ROWS * CELL_SIZE;
    camScale = Math.min(w / worldW, h / worldH) * 0.9;
    camX = (w - worldW * camScale) / 2;
    camY = (h - worldH * camScale) / 2;
  }

  function resizeCanvas() {
    if (!canvas) return;
    var wrap = canvas.parentElement;
    canvas.width = wrap.clientWidth;
    canvas.height = wrap.clientHeight;
    fitView();
    draw();
  }

  // --- drawing helpers ---
  function toScreen(col, row) {
    return {
      x: camX + col * CELL_SIZE * camScale,
      y: camY + row * CELL_SIZE * camScale
    };
  }

  function drawGrid() {
    if (!ctx) return;
    ctx.strokeStyle = 'rgba(255,255,255,0.04)';
    ctx.lineWidth = 0.5;
    var step = Math.max(1, Math.round(10 / camScale));
    for (var c = 0; c <= GRID_COLS; c += step) {
      var p0 = toScreen(c, 0);
      var p1 = toScreen(c, GRID_ROWS);
      ctx.beginPath(); ctx.moveTo(p0.x, p0.y); ctx.lineTo(p1.x, p1.y); ctx.stroke();
    }
    for (var r = 0; r <= GRID_ROWS; r += step) {
      var p0 = toScreen(0, r);
      var p1 = toScreen(GRID_COLS, r);
      ctx.beginPath(); ctx.moveTo(p0.x, p0.y); ctx.lineTo(p1.x, p1.y); ctx.stroke();
    }
  }

  function drawTaxiways() {
    if (!ctx) return;
    var allPaths = (layout.runwayPaths || []).concat(layout.runwayTaxiways || []).concat(layout.taxiways || []);
    allPaths.forEach(function(tw) {
      var verts = tw.vertices;
      if (!verts || verts.length < 2) return;
      var pt = tw.pathType || 'taxiway';
      if (pt === 'runway') {
        ctx.strokeStyle = 'rgba(200,200,200,0.5)';
        ctx.lineWidth = Math.max(2, (tw.width || 15) * camScale * 0.3);
      } else if (pt === 'runway_taxiway') {
        ctx.strokeStyle = 'rgba(180,180,100,0.35)';
        ctx.lineWidth = Math.max(1, (tw.width || 8) * camScale * 0.2);
      } else {
        ctx.strokeStyle = 'rgba(100,160,255,0.25)';
        ctx.lineWidth = Math.max(1, (tw.width || 10) * camScale * 0.15);
      }
      ctx.beginPath();
      var p = toScreen(verts[0].col, verts[0].row);
      ctx.moveTo(p.x, p.y);
      for (var i = 1; i < verts.length; i++) {
        p = toScreen(verts[i].col, verts[i].row);
        ctx.lineTo(p.x, p.y);
      }
      ctx.stroke();
    });
  }

  function drawTerminals() {
    if (!ctx) return;
    (layout.terminals || []).forEach(function(term) {
      var verts = term.vertices;
      if (!verts || verts.length < 3) return;
      ctx.fillStyle = 'rgba(60,60,80,0.35)';
      ctx.strokeStyle = 'rgba(120,120,140,0.3)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      var p = toScreen(verts[0].col, verts[0].row);
      ctx.moveTo(p.x, p.y);
      for (var i = 1; i < verts.length; i++) {
        p = toScreen(verts[i].col, verts[i].row);
        ctx.lineTo(p.x, p.y);
      }
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
    });
  }

  function drawStands() {
    if (!ctx) return;
    var allStands = (layout.pbbStands || []).concat(layout.remoteStands || []);
    allStands.forEach(function(st) {
      var col = st.col; var row = st.row;
      if (col == null || row == null) return;
      var p = toScreen(col, row);
      var r = Math.max(3, 4 * camScale);
      ctx.fillStyle = 'rgba(80,200,120,0.4)';
      ctx.beginPath();
      ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
      ctx.fill();
      if (st.name && camScale > 0.3) {
        ctx.fillStyle = 'rgba(180,220,180,0.6)';
        ctx.font = Math.max(8, 9 * camScale) + 'px sans-serif';
        ctx.fillText(st.name, p.x + r + 2, p.y + 3);
      }
    });
  }

  // --- interpolate flight position at given time ---
  function getFlightPosAtTime(fid, t) {
    var tl = flightTimelines[fid];
    if (!tl || tl.length === 0) return null;
    if (t <= tl[0].t) return { col: tl[0].col, row: tl[0].row };
    if (t >= tl[tl.length - 1].t) return { col: tl[tl.length - 1].col, row: tl[tl.length - 1].row };
    // binary search
    var lo = 0, hi = tl.length - 1;
    while (lo < hi - 1) {
      var mid = (lo + hi) >> 1;
      if (tl[mid].t <= t) lo = mid; else hi = mid;
    }
    var a = tl[lo], b = tl[hi];
    var dt = b.t - a.t;
    if (dt < 0.001) return { col: a.col, row: a.row };
    var frac = (t - a.t) / dt;
    return {
      col: a.col + (b.col - a.col) * frac,
      row: a.row + (b.row - a.row) * frac
    };
  }

  var STATE_COLORS = {
    APPROACH: '#4a90d9',
    LANDING: '#4a90d9',
    TAXI_IN: '#e8c840',
    PARKED: '#3dd68c',
    TAXI_OUT: '#f0a030',
    TAKEOFF: '#f06040',
    DEPARTED: 'rgba(150,150,150,0.3)',
    SCHEDULED: 'rgba(100,100,100,0.2)'
  };

  function guessFlightState(fid, t) {
    var sched = (SIM.schedule || []).find(function(s) { return s.flight_id === fid; });
    if (!sched) return 'TAXI_IN';
    if (sched.ATOT != null && t >= sched.ATOT) return 'DEPARTED';
    if (sched.AOBT != null && t >= sched.AOBT) {
      if (sched.ATOT != null && t < sched.ATOT) return 'TAXI_OUT';
      return 'TAKEOFF';
    }
    if (sched.AIBT != null && t >= sched.AIBT) return 'PARKED';
    if (sched.ALDT != null && t >= sched.ALDT) return 'TAXI_IN';
    if (sched.SLDT != null && t >= sched.SLDT - 120) return 'APPROACH';
    return 'SCHEDULED';
  }

  function drawFlights(t) {
    if (!ctx) return;
    var fids = Object.keys(positions);
    fids.forEach(function(fid) {
      var pos = getFlightPosAtTime(fid, t);
      if (!pos) return;
      var st = guessFlightState(fid, t);
      if (st === 'SCHEDULED' || st === 'DEPARTED') return;
      var color = STATE_COLORS[st] || '#ffffff';
      var p = toScreen(pos.col, pos.row);
      var r = Math.max(4, 5 * camScale);

      // glow
      ctx.fillStyle = color.replace(')', ', 0.2)').replace('rgb', 'rgba').replace('rgba(', 'rgba(');
      var glow = color + '33';
      ctx.beginPath();
      ctx.arc(p.x, p.y, r * 2.5, 0, Math.PI * 2);
      ctx.fillStyle = glow;
      ctx.fill();

      // dot
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
      ctx.fill();

      // label
      if (camScale > 0.2) {
        ctx.fillStyle = '#f0f0f2';
        ctx.font = Math.max(8, 9 * camScale) + 'px sans-serif';
        var label = '';
        var sched = (SIM.schedule || []).find(function(s) { return s.flight_id === fid; });
        if (sched) label = sched.flight_number || sched.reg || fid;
        else label = fid.slice(0, 8);
        ctx.fillText(label, p.x + r + 3, p.y - r + 2);
      }
    });
  }

  // --- main draw ---
  function draw() {
    if (!ctx || !canvas) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawGrid();
    drawTerminals();
    drawTaxiways();
    drawStands();

    var t = allTimes[currentTimeIdx] || 0;
    drawFlights(t);

    if (timeLabel) {
      var totalSec = Math.round(t);
      var hh = Math.floor(totalSec / 3600);
      var mm = Math.floor((totalSec % 3600) / 60);
      var ss = totalSec % 60;
      timeLabel.textContent =
        String(hh).padStart(2, '0') + ':' +
        String(mm).padStart(2, '0') + ':' +
        String(ss).padStart(2, '0');
    }

    if (infoDiv) {
      var activeCount = 0;
      var fids = Object.keys(positions);
      fids.forEach(function(fid) {
        var st = guessFlightState(fid, t);
        if (st !== 'SCHEDULED' && st !== 'DEPARTED') activeCount++;
      });
      infoDiv.textContent = 'Active: ' + activeCount + ' / ' + fids.length + '  |  Frame: ' + (currentTimeIdx + 1) + '/' + totalFrames;
    }
  }

  // --- playback loop ---
  function animLoop(ts) {
    if (!playing) return;
    if (lastFrameTs === 0) lastFrameTs = ts;
    var dt = (ts - lastFrameTs) / 1000.0;
    lastFrameTs = ts;

    if (totalFrames > 1 && allTimes.length > 1) {
      var simTimeDelta = dt * playSpeed;
      var currentSimTime = allTimes[currentTimeIdx] || 0;
      var nextSimTime = currentSimTime + simTimeDelta;
      // find next frame index
      while (currentTimeIdx < totalFrames - 1 && allTimes[currentTimeIdx + 1] <= nextSimTime) {
        currentTimeIdx++;
      }
    }

    if (slider) slider.value = currentTimeIdx;
    draw();

    if (currentTimeIdx >= totalFrames - 1) {
      playing = false;
      return;
    }
    animFrameId = requestAnimationFrame(animLoop);
  }

  function startPlay() {
    if (totalFrames === 0) return;
    playing = true;
    lastFrameTs = 0;
    if (currentTimeIdx >= totalFrames - 1) currentTimeIdx = 0;
    animFrameId = requestAnimationFrame(animLoop);
  }

  function stopPlay() {
    playing = false;
    if (animFrameId) cancelAnimationFrame(animFrameId);
    animFrameId = null;
  }

  // --- controls ---
  if (btnPlay) btnPlay.addEventListener('click', startPlay);
  if (btnPause) btnPause.addEventListener('click', stopPlay);
  if (speedSel) speedSel.addEventListener('change', function() {
    playSpeed = parseFloat(speedSel.value) || 1;
  });
  if (slider) slider.addEventListener('input', function() {
    currentTimeIdx = parseInt(slider.value) || 0;
    draw();
  });

  // --- mouse pan & zoom ---
  var dragging = false;
  var dragStartX = 0, dragStartY = 0;
  var camStartX = 0, camStartY = 0;

  if (canvas) {
    canvas.addEventListener('mousedown', function(e) {
      dragging = true;
      dragStartX = e.clientX;
      dragStartY = e.clientY;
      camStartX = camX;
      camStartY = camY;
    });
    canvas.addEventListener('mousemove', function(e) {
      if (!dragging) return;
      camX = camStartX + (e.clientX - dragStartX);
      camY = camStartY + (e.clientY - dragStartY);
      draw();
    });
    canvas.addEventListener('mouseup', function() { dragging = false; });
    canvas.addEventListener('mouseleave', function() { dragging = false; });
    canvas.addEventListener('wheel', function(e) {
      e.preventDefault();
      var rect = canvas.getBoundingClientRect();
      var mx = e.clientX - rect.left;
      var my = e.clientY - rect.top;
      var factor = e.deltaY < 0 ? 1.1 : 0.9;
      var newScale = camScale * factor;
      newScale = Math.max(0.05, Math.min(20, newScale));
      camX = mx - (mx - camX) * (newScale / camScale);
      camY = my - (my - camY) * (newScale / camScale);
      camScale = newScale;
      draw();
    }, { passive: false });
  }

  // --- init ---
  window.addEventListener('resize', resizeCanvas);
  resizeCanvas();

  if (totalFrames === 0 && infoDiv) {
    infoDiv.textContent = 'No simulation position data available.';
  }
})();
