        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(draft[0][0], draft[0][1]);
        for (let di = 1; di < draft.length; di++) ctx.lineTo(draft[di][0], draft[di][1]);
        if (draft.length >= 2) ctx.stroke();
        ctx.restore();
        draft.forEach(function(pt) {
          ctx.beginPath();
          ctx.arc(pt[0], pt[1], CELL_SIZE * 0.2 * LAYOUT_VERTEX_DOT_SCALE, 0, Math.PI*2);
          ctx.fill();
        });
      }
    }
    ctx.restore();
  }

  function flightTimelineSegmentAtSimTime(flight, tSec) {
    const tl = flight && flight.timeline;
    if (!tl || tl.length < 2) return null;
    let t = Number(tSec);
    if (!isFinite(t)) return null;
    if (t + 1e-9 < tl[0].t) return null;
    if (t > tl[tl.length - 1].t) t = tl[tl.length - 1].t;
    for (let i = 0; i < tl.length - 1; i++) {
      const a = tl[i], b = tl[i + 1];
      if (t >= a.t && t <= b.t) return { a: a, b: b };
    }
    return null;
  }
  function isTimelineSegmentStationaryWorld(a, b) {
    const dx = b.x - a.x, dy = b.y - a.y;
    return dx * dx + dy * dy < 0.64;
  }
  function countFlightsWaitingAtHoldingPoint2D(hp, tSec) {
    if (!hp || !isFinite(hp.x) || !isFinite(hp.y)) return 0;
    if (!state.hasSimulationResult || !state.globalUpdateFresh) return 0;
    if (typeof getFlightPoseAtTimeForDraw !== 'function') return 0;
    const t = Number(tSec);
    if (!isFinite(t)) return 0;
    const hx = hp.x, hy = hp.y;
    const dia = typeof c2dHoldingPointDiameterM === 'function' ? c2dHoldingPointDiameterM() : 24;
    const rad = Math.max(10, dia * 0.55);
    const rad2 = rad * rad;
    let n = 0;
    const flights = state.flights || [];
    for (let i = 0; i < flights.length; i++) {
      const f = flights[i];
      if (!f || flightBlockedLikeNoWay(f)) continue;
      const pose = getFlightPoseAtTimeForDraw(f, t);
      if (!pose) continue;
      const dx = pose.x - hx, dy = pose.y - hy;
      if (dx * dx + dy * dy > rad2) continue;
      const seg = flightTimelineSegmentAtSimTime(f, t);
      if (!seg || !isTimelineSegmentStationaryWorld(seg.a, seg.b)) continue;
      n++;
    }
    return n;
  }
  function firstFlightWaitingAtHoldingPoint2D(hp, tSec) {
    if (!hp || !isFinite(hp.x) || !isFinite(hp.y)) return null;
    if (!state.hasSimulationResult || !state.globalUpdateFresh) return null;
    if (typeof getFlightPoseAtTimeForDraw !== 'function') return null;
    const t = Number(tSec);
    if (!isFinite(t)) return null;
    const hx = hp.x, hy = hp.y;
    const dia = typeof c2dHoldingPointDiameterM === 'function' ? c2dHoldingPointDiameterM() : 24;
    const rad = Math.max(10, dia * 0.55);
    const rad2 = rad * rad;
    const flights = state.flights || [];
    for (let i = 0; i < flights.length; i++) {
      const f = flights[i];
      if (!f || flightBlockedLikeNoWay(f)) continue;
      const pose = getFlightPoseAtTimeForDraw(f, t);
      if (!pose) continue;
      const dx = pose.x - hx, dy = pose.y - hy;
      if (dx * dx + dy * dy > rad2) continue;
      const seg = flightTimelineSegmentAtSimTime(f, t);
      if (!seg || !isTimelineSegmentStationaryWorld(seg.a, seg.b)) continue;
      return f;
    }
    return null;
  }
  function polylineTangentForwardAtDistance(pts, sAlong) {
    if (!pts || pts.length < 2) return [1, 0];
    if (typeof polylineTotalLength !== 'function' || typeof polylinePointAtDistance !== 'function') return [1, 0];
    const total = polylineTotalLength(pts);
    if (total < 1e-6) return [1, 0];
    const eps = 2;
    const s0 = Math.max(0, Math.min(Number(sAlong) || 0, total));
    let s1 = Math.min(s0 + eps, total);
    let p0 = polylinePointAtDistance(pts, s0);
    let p1 = polylinePointAtDistance(pts, s1);
    let dx = p1[0] - p0[0], dy = p1[1] - p0[1];
    if (dx * dx + dy * dy < 1e-10) {
      s1 = Math.max(0, s0 - eps);
      p1 = polylinePointAtDistance(pts, s0);
      p0 = polylinePointAtDistance(pts, s1);
      dx = p1[0] - p0[0];
      dy = p1[1] - p0[1];
    }
    const len = Math.hypot(dx, dy) || 1;
    return [dx / len, dy / len];
  }
  function drawHoldingQueueGhostFlights2D() {
    if (!ctx) return;
    if (!state.hasSimulationResult || !state.globalUpdateFresh) return;
    if (!state.flights || !state.flights.length) return;
    if (typeof getFlightPoseAtTimeForDraw !== 'function') return;
    if (typeof graphPathDeparture !== 'function' || typeof cumulativeDistAlongPolylineToPoint !== 'function') return;
    if (typeof polylinePointAtDistance !== 'function' || typeof polylineTotalLength !== 'function') return;
    const tSecDraw = state.simTimeSec;
    if (typeof prepareLazyTimelinesForCurrentSim === 'function') prepareLazyTimelinesForCurrentSim(tSecDraw);
    const HOLDING_QUEUE_GHOST_SPACING_M = 70;
    const dia = typeof c2dHoldingPointDiameterM === 'function' ? c2dHoldingPointDiameterM() : 24;
    const rad = Math.max(10, dia * 0.55);
    const pathTol2 = Math.pow(Math.max(rad * 4, 45), 2);
    const silN = Number(_acSil.noseX), silWR = Number(_acSil.wingRearX), silUY = Number(_acSil.wingUpperY);
    const silTN = Number(_acSil.tailNeckX), silLY = Number(_acSil.wingLowerY);
    const nX = isFinite(silN) ? silN : 0.6;
    const wRx = isFinite(silWR) ? silWR : -0.5;
    const uY = isFinite(silUY) ? silUY : 0.35;
    const tX = isFinite(silTN) ? silTN : -0.3;
    const lY = isFinite(silLY) ? silLY : -0.35;
    const useDetailSil = _ac2d.useDetailedSilhouette === true;
    const silhouette2D = [
      [0.86, 0],
      [0.74, 0.038], [0.55, 0.046], [0.35, 0.048], [0.16, 0.05],
      [-0.16, 0.5],
      [-0.22, 0.5],
      [-0.38, 0.09], [-0.52, 0.056], [-0.66, 0.046],
      [-0.76, 0.15],
      [-0.82, 0.036], [-0.88, 0],
      [-0.82, -0.036],
      [-0.76, -0.15],
      [-0.66, -0.046], [-0.52, -0.056], [-0.38, -0.09],
      [-0.22, -0.5],
      [-0.16, -0.5],
      [0.16, -0.05], [0.35, -0.048], [0.55, -0.046], [0.74, -0.038],
    ];
    let scaleX, scaleY;
    if (useDetailSil) {
      let minXn = Infinity, maxXn = -Infinity, maxYy = 0;
      for (let si = 0; si < silhouette2D.length; si++) {
        const px = silhouette2D[si][0], py = silhouette2D[si][1];
        minXn = Math.min(minXn, px);
        maxXn = Math.max(maxXn, px);
        maxYy = Math.max(maxYy, Math.abs(py));
      }
      const lenNorm = Math.max(1e-9, maxXn - minXn);
      const wingNorm = Math.max(1e-9, 2 * maxYy);
      scaleX = AIRCRAFT_FUSELAGE_LENGTH_M / lenNorm;
      scaleY = AIRCRAFT_WINGSPAN_M / wingNorm;
    } else {
      const xs = [nX, wRx, tX];
      const minXn = Math.min(xs[0], xs[1], xs[2]);
      const maxXn = Math.max(xs[0], xs[1], xs[2]);
      const lenNorm = Math.max(1e-9, maxXn - minXn);
      const wingNorm = Math.max(1e-9, uY + lY);
      scaleX = AIRCRAFT_FUSELAGE_LENGTH_M / lenNorm;
      scaleY = AIRCRAFT_WINGSPAN_M / wingNorm;
    }
    const outW = Number(_ac2d.outlineWidth);
    const outlineWidth = (isFinite(outW) && outW > 0) ? outW : 0;
    const outlineColor = _ac2d.outlineColor || '';
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.translate(state.panX, state.panY);
    ctx.scale(state.scale, state.scale);
    (state.holdingPoints || []).forEach(function(hp) {
      if (!hp || !isFinite(hp.x) || !isFinite(hp.y)) return;
      const waitN = countFlightsWaitingAtHoldingPoint2D(hp, tSecDraw);
      if (waitN < 2) return;
      const f = firstFlightWaitingAtHoldingPoint2D(hp, tSecDraw);
      if (!f) return;
      const pts = graphPathDeparture(f, { onlyToLineup: true });
      if (!pts || pts.length < 2) return;
      const cum = cumulativeDistAlongPolylineToPoint(pts, [hp.x, hp.y]);
      if (!cum || cum.d2 > pathTol2) return;
      const sHp = cum.distAlong;
      for (let k = 1; k < waitN; k++) {
        const s = sHp - k * HOLDING_QUEUE_GHOST_SPACING_M;
        if (s < -0.5) break;
        const sDraw = Math.max(0, s);
        const pt = polylinePointAtDistance(pts, sDraw);
        const tan = polylineTangentForwardAtDistance(pts, sDraw);
        const nx = tan[0], ny = tan[1];
        ctx.save();
        ctx.translate(pt[0], pt[1]);
        ctx.rotate(Math.atan2(ny, nx));
        ctx.fillStyle = apron2DGlyphFill();
        ctx.beginPath();
        if (useDetailSil) {
          ctx.moveTo(silhouette2D[0][0] * scaleX, silhouette2D[0][1] * scaleY);
          for (let si = 1; si < silhouette2D.length; si++) ctx.lineTo(silhouette2D[si][0] * scaleX, silhouette2D[si][1] * scaleY);
          ctx.closePath();
        } else {
          ctx.moveTo(scaleX * nX, 0);
          ctx.lineTo(scaleX * wRx, scaleY * uY);
          ctx.lineTo(scaleX * tX, 0);
          ctx.lineTo(scaleX * wRx, scaleY * lY);
          ctx.closePath();
        }
        ctx.fill();
        if (outlineWidth > 0 && outlineColor) {
          ctx.strokeStyle = outlineColor;
          ctx.lineWidth = outlineWidth;
          ctx.stroke();
        } else if (useDetailSil) {
          ctx.strokeStyle = 'rgba(15,23,42,1)';
          ctx.lineWidth = 1.1;
          ctx.stroke();
        }
        ctx.restore();
      }
    });
    ctx.restore();
  }
  function drawHoldingPoints2D() {
    if (!ctx) return;
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.translate(state.panX, state.panY);
    ctx.scale(state.scale, state.scale);
    const r = c2dHoldingPointDiameterM() * 0.5;
    const sel = state.selectedObject && state.selectedObject.type === 'holdingPoint';
    (state.holdingPoints || []).forEach(function(hp) {
      if (!hp || !isFinite(hp.x) || !isFinite(hp.y)) return;
      const selected = sel && state.selectedObject.id === hp.id;
      const k = normalizeHoldingPointKind(hp.hpKind);
      const fill = c2dHoldingPointFillForKind(k);
      const stroke = c2dHoldingPointStrokeForKind(k);
      ctx.beginPath();
      ctx.arc(hp.x, hp.y, r, 0, Math.PI * 2);
      ctx.fillStyle = selected ? c2dObjectSelectedFill() : fill;
      ctx.strokeStyle = selected ? c2dObjectSelectedStroke() : stroke;
      ctx.lineWidth = selected ? 2.5 : 1;
      if (selected) {
        ctx.shadowColor = c2dObjectSelectedGlow();
        ctx.shadowBlur = c2dObjectSelectedGlowBlur();
      } else {
        ctx.shadowBlur = 0;
      }
      ctx.fill();
      if (!selected) ctx.stroke();
      ctx.shadowBlur = 0;
      const waitN = countFlightsWaitingAtHoldingPoint2D(hp, state.simTimeSec);
      if (waitN > 0) {
        const bx = hp.x + r * 1.05 + 6;
        const by = hp.y - r * 1.05;
        const label = String(waitN);
        const fs = Math.max(9, Math.min(15, 11 / Math.max(0.22, state.scale)));
        ctx.font = 'bold ' + fs + 'px system-ui, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        const tw = ctx.measureText(label).width;
        const padX = fs * 0.42;
        const padY = fs * 0.28;
        const bw = tw + padX * 2;
        const bh = fs + padY * 2;
        const left = bx - bw / 2;
        const top = by - bh / 2;
        const rr = Math.min(bh * 0.45, fs * 0.5);
        ctx.beginPath();
        ctx.moveTo(left + rr, top);
        ctx.lineTo(left + bw - rr, top);
        ctx.quadraticCurveTo(left + bw, top, left + bw, top + rr);
        ctx.lineTo(left + bw, top + bh - rr);
        ctx.quadraticCurveTo(left + bw, top + bh, left + bw - rr, top + bh);
        ctx.lineTo(left + rr, top + bh);
        ctx.quadraticCurveTo(left, top + bh, left, top + bh - rr);
        ctx.lineTo(left, top + rr);
        ctx.quadraticCurveTo(left, top, left + rr, top);
        ctx.closePath();
        ctx.fillStyle = 'rgba(15, 23, 42, 0.94)';
        ctx.strokeStyle = 'rgba(148, 163, 184, 0.95)';
        ctx.lineWidth = Math.max(0.75, 1.15 / Math.max(state.scale, 0.08));
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = '#f1f5f9';
        ctx.fillText(label, bx, by);
      }
    });
    if (state.holdingPointDrawing && state.previewHoldingPoint) {
      const px = state.previewHoldingPoint.x, py = state.previewHoldingPoint.y;
      const ptp = state.previewHoldingPoint.pathType || 'taxiway';
      ctx.beginPath();
      ctx.arc(px, py, r, 0, Math.PI * 2);
      ctx.fillStyle = c2dHoldingPointPreviewFillForPathType(ptp);
      ctx.strokeStyle = c2dHoldingPointPreviewStrokeForPathType(ptp);
      ctx.lineWidth = 1;
      ctx.shadowBlur = 0;
      ctx.fill();
    }
    ctx.restore();
  }

  function drawStandPreview() {
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.translate(state.panX, state.panY);
    ctx.scale(state.scale, state.scale);
    const mode = settingModeSelect.value;
    if (mode === 'remote' && state.previewRemote) {
      const cx = Number(state.previewRemote.x), cy = Number(state.previewRemote.y);
      const category = document.getElementById('remoteCategory').value || 'C';
      const size = getStandSizeMeters(category);
      const angle = normalizeAngleDeg(document.getElementById('remoteAngle') ? document.getElementById('remoteAngle').value : 0) * Math.PI / 180;
      const overlap = state.previewRemote.overlap;
      ctx.fillStyle = overlap ? 'rgba(239,68,68,0.35)' : 'rgba(34,197,94,0.25)';
      ctx.strokeStyle = overlap ? '#ef4444' : '#22c55e';
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 4]);
      ctx.save();
      ctx.translate(cx, cy);
      ctx.rotate(angle);
      ctx.beginPath();
      ctx.rect(-size/2, -size/2, size, size);
      ctx.fill();
      ctx.stroke();
      ctx.restore();
    }
    if (mode === 'pbb' && state.previewPbb) {
      const ex = state.previewPbb.x2, ey = state.previewPbb.y2;
      const size = getStandSizeMeters(state.previewPbb.category || 'C');
      const overlap = state.previewPbb.overlap;
      const angle = getPBBStandAngle(state.previewPbb);
      ctx.fillStyle = overlap ? 'rgba(239,68,68,0.35)' : 'rgba(34,197,94,0.25)';
      ctx.strokeStyle = overlap ? '#ef4444' : '#22c55e';
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 4]);
      ctx.save();
      ctx.translate(ex, ey);
      ctx.rotate(angle);
      ctx.beginPath();
      ctx.rect(-size/2, -size/2, size, size);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = '#bbf7d0';
      ctx.font = '10px system-ui';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(state.previewPbb.category || document.getElementById('standCategory').value || 'C', 0, 0);
      ctx.restore();
    }
    ctx.restore();
  }

  let _safeDrawErrLogged = false;
  let _drawRafId = 0;
  function safeDraw() { try { draw(); _safeDrawErrLogged = false; } catch(e) { if (!_safeDrawErrLogged) { console.error('safeDraw: draw() error', e); _safeDrawErrLogged = true; } } }
  function flushDrawNow() {
    if (_drawRafId) {
      cancelAnimationFrame(_drawRafId);
      _drawRafId = 0;
    }
    safeDraw();
  }
  function scheduleDraw() {
    if (_drawRafId) return;
    _drawRafId = requestAnimationFrame(function() {
      _drawRafId = 0;
      safeDraw();
    });
  }
  function draw() {
    if (!ctx || !canvas) return;
    if (state.simSliderScrubbing) return;
    drawGrid();
    drawTerminals();
    drawTaxiways();
    drawHoldingPoints2D();
    drawPBBs();
    drawRemoteStands();
    drawApronTaxiwayLinks();
    drawStandPreview();
    drawSelectedLayoutEdge();
    {
      const sel = state.selectedObject;
      const rid = state.flightPathRevealFlightId;
      if (sel && sel.type === 'flight' && rid != null && sel.id === rid) {
        drawFlightPathHighlight();
        drawDeparturePathHighlight();
      }
    }
    drawApproachPreviewPaths2D();
    drawHoldingQueueGhostFlights2D();
    drawFlights2D();
    drawPathJunctions();
  }

  document.addEventListener('keydown', function(ev) {
    const el = document.activeElement;
    const inInput = el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.isContentEditable);
    if (ev.ctrlKey && ev.key === 'z') {
      if (!inInput) { ev.preventDefault(); undo(); }
      return;
    }
    if (ev.key === 'Escape') {
      if (inInput) return;
      const anyLayoutDraw = !!(state.pbbDrawing || state.remoteDrawing || state.holdingPointDrawing || state.apronLinkDrawing ||
        state.terminalDrawingId || state.taxiwayDrawingId);
      if (!anyLayoutDraw) return;
      ev.preventDefault();
      cancelActiveLayoutDrawingState();
      state.terminalDrawingId = null;
      state.taxiwayDrawingId = null;
      syncPanelFromState();
      updateObjectInfo();
      if (typeof redrawLayoutAfterEdit === 'function') redrawLayoutAfterEdit();
      else if (typeof updateAllFlightPaths === 'function') updateAllFlightPaths(); else draw();
      return;
    }
    if (ev.key !== 'Delete' && ev.key !== 'Backspace') return;
    if (inInput) return;
    if (removeLastDrawingVertex()) {
      ev.preventDefault();
      return;
    }
    if (removeSelectedVertex()) {
      ev.preventDefault();
      return;
    }
    if (!state.selectedObject) return;
    const type = state.selectedObject.type;
    const id = state.selectedObject.id;
    if (type !== 'terminal' && type !== 'pbb' && type !== 'remote' && type !== 'holdingPoint' && type !== 'taxiway' && type !== 'apronLink' && type !== 'flight') return;
    pushUndo();
    removeLayoutObjectFromState(type, id);
    state.selectedObject = null;
    state.selectedVertex = null;
    if (type === 'terminal' && state.currentTerminalId === id) {
      state.currentTerminalId = state.terminals.length ? state.terminals[0].id : null;
      if (state.terminalDrawingId === id) {
        state.terminalDrawingId = null;
        state.layoutPathDrawPointer = null;
      }
    }
    if (type === 'taxiway' && state.taxiwayDrawingId === id) {
      state.taxiwayDrawingId = null;
      state.layoutPathDrawPointer = null;
    }
    syncPanelFromState();
    updateObjectInfo();
    if (typeof redrawLayoutAfterEdit === 'function') redrawLayoutAfterEdit();
    else if (typeof updateAllFlightPaths === 'function') updateAllFlightPaths(); else draw();
    ev.preventDefault();
  });

  container.addEventListener('mousedown', function(ev) {
    if (ev.button !== 0) return;
    const rect = canvas.getBoundingClientRect();
    const sx = ev.clientX - rect.left, sy = ev.clientY - rect.top;
    const [wx, wy] = screenToWorld(sx, sy);
    const mode = settingModeSelect.value;
    if (mode === 'terminal' && !state.terminalDrawingId) {
      const vhit = hitTestTerminalVertex(wx, wy);
      if (vhit) {
        pushUndo();
        state.dragVertex = vhit;
        state.selectedVertex = { type: 'terminal', id: vhit.terminalId, index: vhit.index };
        const term = state.terminals.find(t => t.id === vhit.terminalId);
        if (term) {
          state.flightPathRevealFlightId = null;
          state.selectedObject = { type: 'terminal', id: term.id, obj: term };
          state.currentTerminalId = term.id;
          syncPanelFromState();
          updateObjectInfo();
          draw();
        }
        return;
      }
    }
    if (state.selectedObject && state.selectedObject.type === 'taxiway') {
      const thit = hitTestTaxiwayVertex(wx, wy);
      if (thit && thit.taxiwayId === state.selectedObject.id) {
        pushUndo();
        state.dragTaxiwayVertex = thit;
        state.selectedVertex = { type: 'taxiway', id: thit.taxiwayId, index: thit.index };
        draw();
        return;
      }
    }
    const standRotateHit = hitTestStandRotationHandle(wx, wy);
    if (standRotateHit) {
      pushUndo();
      state.dragStandRotation = standRotateHit;
      state.selectedVertex = { type: 'standRotation', id: standRotateHit.id, standType: standRotateHit.type };
      draw();
      return;
    }
    if (state.selectedObject && state.selectedObject.type === 'pbb' && !state.pbbDrawing) {
      const ph = hitTestPbbEditablePoint(wx, wy);
      if (ph) {
        pushUndo();
        if (ph.type === 'bridge') {
          state.dragPbbBridgeVertex = { pbbId: state.selectedObject.id, bridgeIndex: ph.bridgeIndex, pointIndex: ph.pointIndex };
          state.selectedVertex = { type: 'pbbBridge', id: state.selectedObject.id, bridgeIndex: ph.bridgeIndex, pointIndex: ph.pointIndex };
        } else {


          state.dragStandConnection = { pbbId: state.selectedObject.id };
          state.selectedVertex = { type: 'pbbApronSite', id: state.selectedObject.id };
        }
        draw();
        return;
      }
    }
    if (state.selectedObject && state.selectedObject.type === 'apronLink' && !state.apronLinkDrawing) {
      const ah = hitTestApronLinkVertex(wx, wy);
      if (ah && ah.linkId === state.selectedObject.id) {
        pushUndo();
        state.dragApronLinkVertex = ah;
        state.selectedVertex = ah.kind === 'mid'
          ? { type: 'apronLink', id: ah.linkId, kind: 'mid', midIndex: ah.midIndex }
          : { type: 'apronLink', id: ah.linkId, kind: 'taxiway' };
        draw();
        return;
      }
    }
    state.selectedVertex = null;
    if ((mode === 'pbb' && state.pbbDrawing) || (mode === 'remote' && state.remoteDrawing) || (mode === 'holdingPoint' && state.holdingPointDrawing)) return;
    state.dragStart = { sx, sy, panX: state.panX, panY: state.panY };
    state.isPanning = false;
  });
  container.addEventListener('mousemove', function(ev) {
    const rect = canvas.getBoundingClientRect();
    const sx = ev.clientX - rect.left, sy = ev.clientY - rect.top;
    const [wx, wy] = screenToWorld(sx, sy);
    const snappedPt = worldPointToCellPoint(wx, wy, !!ev.shiftKey);
    const snappedPx = cellToPixel(snappedPt.col, snappedPt.row);
    const [col, row] = pixelToCell(wx, wy);
    if (coordEl) coordEl.textContent = 'cell: (' + col + ', ' + row + ')';
    const prev = state.hoverCell;
    state.hoverCell = { col, row };
    const hoverChanged = !prev || prev.col !== col || prev.row !== row;
    let drewThisMove = false;
    if (settingModeSelect.value === 'apronTaxiway' && state.apronLinkDrawing && state.apronLinkTemp) {
      const pw = state.apronLinkPointerWorld;
      if (!pw || pw[0] !== wx || pw[1] !== wy) {
        state.apronLinkPointerWorld = [wx, wy];
        scheduleDraw(); drewThisMove = true;
      }
    } else if (state.apronLinkPointerWorld) {
      state.apronLinkPointerWorld = null;
      scheduleDraw(); drewThisMove = true;
    }
    const pathLayoutDrawing = !!(state.terminalDrawingId || state.taxiwayDrawingId);
    const blockLayoutPathPtr = !!(state.isPanning || state.dragVertex || state.dragTaxiwayVertex || state.dragPbbBridgeVertex || state.dragStandConnection || state.dragApronLinkVertex || state.dragStandRotation);
    if (pathLayoutDrawing && !blockLayoutPathPtr) {
      const nx = snappedPx[0], ny = snappedPx[1];
      const lp = state.layoutPathDrawPointer;
      if (!lp || lp[0] !== nx || lp[1] !== ny) {
        state.layoutPathDrawPointer = [nx, ny];
        scheduleDraw(); drewThisMove = true;
      }
    } else if (state.layoutPathDrawPointer && (!pathLayoutDrawing || blockLayoutPathPtr)) {
      state.layoutPathDrawPointer = null;
      if (!drewThisMove) { scheduleDraw(); drewThisMove = true; }
    }
    if (state.dragVertex) {
      const term = state.terminals.find(t => t.id === state.dragVertex.terminalId);
      if (term && term.vertices[state.dragVertex.index]) {
        const v = term.vertices[state.dragVertex.index];
        v.col = snappedPt.col;
        v.row = snappedPt.row;
        scheduleDraw(); drewThisMove = true;
      }
      return;
    }
    if (state.dragTaxiwayVertex) {
      const tw = state.taxiways.find(t => t.id === state.dragTaxiwayVertex.taxiwayId);
      if (tw && tw.vertices[state.dragTaxiwayVertex.index]) {
        const v = tw.vertices[state.dragTaxiwayVertex.index];
        v.col = snappedPt.col;
        v.row = snappedPt.row;
        scheduleDraw(); drewThisMove = true;
        if (scene3d) update3DScene();
      }
      return;
    }
    if (state.dragStandRotation) {
      if (state.dragStandRotation.type === 'pbb') {
        const pbb = state.pbbStands.find(function(item) { return item.id === state.dragStandRotation.id; });
        if (pbb) {
          const origin = getPbbRotationOriginPx(pbb);
          const nextDeg = normalizeAngleDeg(Math.atan2(wy - origin[1], wx - origin[0]) * 180 / Math.PI);
          setPbbGeometryFromAngleLength(pbb, nextDeg, getPbbLengthMeters(pbb), true);
          const angleInput = document.getElementById('standAngle');
          if (angleInput) angleInput.value = String(Math.round(getPbbAngleDeg(pbb)));
          scheduleDraw(); drewThisMove = true;
          if (scene3d) update3DScene();
        }
      } else if (state.dragStandRotation.type === 'remote') {
        const st = state.remoteStands.find(function(item) { return item.id === state.dragStandRotation.id; });
        if (st) {
          const center = getRemoteStandCenterPx(st);
          const nextDeg = normalizeAngleDeg(Math.atan2(wy - center[1], wx - center[0]) * 180 / Math.PI);
          st.angleDeg = nextDeg;
          const angleInput = document.getElementById('remoteAngle');
          if (angleInput) angleInput.value = String(Math.round(nextDeg));
          scheduleDraw(); drewThisMove = true;
          if (scene3d) update3DScene();
        }
      }
      return;
    }
    if (state.dragPbbBridgeVertex) {
      const pbb = state.pbbStands.find(function(item) { return item.id === state.dragPbbBridgeVertex.pbbId; });
      if (pbb && Array.isArray(pbb.pbbBridges) && pbb.pbbBridges[state.dragPbbBridgeVertex.bridgeIndex] && Array.isArray(pbb.pbbBridges[state.dragPbbBridgeVertex.bridgeIndex].points) && pbb.pbbBridges[state.dragPbbBridgeVertex.bridgeIndex].points[state.dragPbbBridgeVertex.pointIndex]) {
        const pt = pbb.pbbBridges[state.dragPbbBridgeVertex.bridgeIndex].points[state.dragPbbBridgeVertex.pointIndex];
        if (state.dragPbbBridgeVertex.pointIndex === 0) {
          const projected = getClosestTerminalEdgePoint(wx, wy);
          if (projected && projected.point) {
            pt.x = projected.point[0];
            pt.y = projected.point[1];
          }
        } else {
          pt.x = snappedPx[0];
          pt.y = snappedPx[1];
        }
        scheduleDraw(); drewThisMove = true;
        if (scene3d) update3DScene();
      }
      return;
    }
    if (state.dragStandConnection) {
      const pbb = state.pbbStands.find(function(item) { return item.id === state.dragStandConnection.pbbId; });
      if (pbb) {
        pbb.apronSiteX = snappedPx[0];
        pbb.apronSiteY = snappedPx[1];
        scheduleDraw(); drewThisMove = true;
        if (scene3d) update3DScene();
      }
      return;
    }
    if (state.dragApronLinkVertex) {
      const lk = state.apronLinks.find(l => l.id === state.dragApronLinkVertex.linkId);
      if (!lk) {
        state.dragApronLinkVertex = null;
      } else if (state.dragApronLinkVertex.kind === 'mid') {
        const mi = state.dragApronLinkVertex.midIndex;
        if (lk.midVertices && mi >= 0 && mi < lk.midVertices.length &&
            col >= 0 && row >= 0 && col <= GRID_COLS && row <= GRID_ROWS) {
          lk.midVertices[mi].col = snappedPt.col;
          lk.midVertices[mi].row = snappedPt.row;
          scheduleDraw(); drewThisMove = true;
          if (scene3d) update3DScene();
        }
      } else if (state.dragApronLinkVertex.kind === 'taxiway') {
        const snap = snapWorldPointToTaxiwayPolyline(wx, wy, lk.taxiwayId);
        if (snap) {
          lk.tx = snap[0];
          lk.ty = snap[1];
          scheduleDraw(); drewThisMove = true;
          if (scene3d) update3DScene();
        }
      }
      return;
    }
    if (state.dragStart) {
      const dx = sx - state.dragStart.sx, dy = sy - state.dragStart.sy;
      if (!state.isPanning && (Math.abs(dx) > DRAG_THRESH || Math.abs(dy) > DRAG_THRESH))
        state.isPanning = true;
      if (state.isPanning) {
        state.panX = state.dragStart.panX + dx;
        state.panY = state.dragStart.panY + dy;
        scheduleDraw(); drewThisMove = true;
      }
    }
    const mode = settingModeSelect.value;
    if (!state.isPanning && !state.dragVertex && mode === 'holdingPoint' && state.holdingPointDrawing) {
      const snap = snapHoldingPointOnAllowedTaxiways(wx, wy);
      if (snap) {
        state.previewHoldingPoint = { x: snap.x, y: snap.y, pathType: snap.pathType };
      } else {
        state.previewHoldingPoint = null;
      }
      scheduleDraw(); drewThisMove = true;
    } else if (!state.isPanning && !state.dragVertex && mode === 'remote' && state.remoteDrawing) {
      const category = document.getElementById('remoteCategory').value || 'C';
      const angleDeg = normalizeAngleDeg(document.getElementById('remoteAngle') ? document.getElementById('remoteAngle').value : 0);
      const candidate = { x: snappedPx[0], y: snappedPx[1], category, angleDeg };
      const candCorners = getRemoteStandCorners(candidate);
      let overlap = false;
      for (let i = 0; i < state.remoteStands.length; i++) {
        if (rotatedRectsOverlap(candCorners, getRemoteStandCorners(state.remoteStands[i]))) { overlap = true; break; }
      }
      if (!overlap) {
        for (let i = 0; i < state.pbbStands.length; i++) {
          if (rotatedRectsOverlap(candCorners, getPBBStandCorners(state.pbbStands[i]))) { overlap = true; break; }
        }
      }
      const maxX = GRID_COLS * CELL_SIZE, maxY = GRID_ROWS * CELL_SIZE;
      if (candidate.x < 0 || candidate.y < 0 || candidate.x > maxX || candidate.y > maxY) overlap = true;
      state.previewRemote = { x: candidate.x, y: candidate.y, overlap };
      scheduleDraw(); drewThisMove = true;
    } else if (!state.isPanning && !state.dragVertex && mode === 'pbb' && state.pbbDrawing) {
      let bestEdge = null, bestD2 = Infinity;
      state.terminals.forEach(t => {
        if (!t.closed || t.vertices.length < 2) return;
        for (let i = 0; i < t.vertices.length; i++) {
          const v1 = t.vertices[i], v2 = t.vertices[(i+1) % t.vertices.length];
          const p1 = cellToPixel(v1.col, v1.row), p2 = cellToPixel(v2.col, v2.row);
          const near = closestPointOnSegment(p1, p2, snappedPx);
          if (near) {
            const d2 = dist2(near, snappedPx);
            if (d2 < bestD2) { bestD2 = d2; bestEdge = { near, p1, p2 }; }
          }
        }
      });
      const maxD2 = (CELL_SIZE*1.0)**2;
      if (bestEdge && bestD2 < maxD2) {
        const nearPt = bestEdge.near;
        const ex = (nearPt && nearPt[0] != null) ? nearPt[0] : 0;
        const ey = (nearPt && nearPt[1] != null) ? nearPt[1] : 0;
        const [x1,y1]=bestEdge.p1, [x2,y2]=bestEdge.p2;
        let nx = -(y2-y1), ny = x2-x1;
        const len = Math.hypot(nx,ny) || 1; nx /= len; ny /= len;
        const toClickX = snappedPx[0] - ex, toClickY = snappedPx[1] - ey;
        if (nx * toClickX + ny * toClickY < 0) { nx *= -1; ny *= -1; }
        const category = document.getElementById('standCategory').value || 'C';
        const standSize = getStandSizeMeters(category);
        const minLen = standSize / 2 + 3;
        const lenMeters = Number(document.getElementById('pbbLength').value || 15);
        const lenPx = Math.max(isFinite(lenMeters) && lenMeters > 0 ? lenMeters : 15, minLen);
        const px2 = ex + nx * lenPx, py2 = ey + ny * lenPx;
        const preview = { x1: ex, y1: ey, x2: px2, y2: py2, category };
        const overlap = pbbStandOverlapsExisting(preview);
        state.previewPbb = { x1: ex, y1: ey, x2: px2, y2: py2, category: preview.category, overlap };
        scheduleDraw(); drewThisMove = true;
      } else {
        if (state.previewPbb) { state.previewPbb = null; scheduleDraw(); drewThisMove = true; }
      }
    } else {
      let clearedPreview = false;
      if (state.previewRemote) { state.previewRemote = null; clearedPreview = true; }
      if (state.previewPbb) { state.previewPbb = null; clearedPreview = true; }
      if (state.previewHoldingPoint) { state.previewHoldingPoint = null; clearedPreview = true; }
      if (clearedPreview) { scheduleDraw(); drewThisMove = true; }
    }
    if (flightTooltip && !state.isPanning) {
      let tipDone = false;
      if (state.hasSimulationResult && state.globalUpdateFresh) {
        let bestFlight = null;
        let bestD2 = (CELL_SIZE * FLIGHT_TOOLTIP_CF) ** 2;
        const tSec = state.simTimeSec;
        if (typeof prepareLazyTimelinesForCurrentSim === 'function') prepareLazyTimelinesForCurrentSim(tSec);
        state.flights.forEach(f => {
          const pose = getFlightPoseAtTimeForDraw(f, tSec);
          if (!pose || f.reg == null || !String(f.reg).trim()) return;
          const dx = pose.x - wx;
          const dy = pose.y - wy;
          const d2 = dx * dx + dy * dy;
          if (d2 < bestD2) { bestD2 = d2; bestFlight = f; }
        });
        if (bestFlight && bestFlight.reg) {
          flightTooltip.style.display = 'block';
          flightTooltip.textContent = String(bestFlight.reg).trim();
          flightTooltip.style.left = (ev.clientX + 12) + 'px';
          flightTooltip.style.top = (ev.clientY + 12) + 'px';
          tipDone = true;
        }
      }
      if (!tipDone) {
        const hit = hitTest(wx, wy);
        if (hit && hit.obj) {
          const name = (hit.obj.name != null && String(hit.obj.name).trim()) ? String(hit.obj.name).trim() : (hit.type === 'terminal' ? 'Building' : hit.type === 'pbb' ? 'Contact Stand' : hit.type === 'remote' ? 'Remote Stand' : hit.type === 'holdingPoint' ? holdingPointKindDisplayLabel(hit.obj.hpKind) : hit.type === 'taxiway' ? (hit.obj.name || 'Path') : hit.type === 'apronLink' ? (hit.obj.name || 'Apron Taxiway') : hit.type);
          flightTooltip.style.display = 'block';
          flightTooltip.textContent = name;
          flightTooltip.style.left = (ev.clientX + 12) + 'px';
          flightTooltip.style.top = (ev.clientY + 12) + 'px';
        } else {
          flightTooltip.style.display = 'none';
        }
      }
    }
    if (hoverChanged && !drewThisMove) { scheduleDraw(); drewThisMove = true; }
  });
  container.addEventListener('mouseleave', function() {
    state.dragStart = null;
    state.isPanning = false;
    state.dragStandRotation = null;
    state.dragPbbBridgeVertex = null;
    state.dragStandConnection = null;
    state.hoverCell = null;
    state.previewPbb = null;
    state.previewRemote = null;
    state.previewHoldingPoint = null;
    state.apronLinkPointerWorld = null;
    flushDrawNow();
  });
  container.addEventListener('dblclick', function(ev) {
    if (ev.button !== 0) return;
    const rect = canvas.getBoundingClientRect();
    const sx = ev.clientX - rect.left, sy = ev.clientY - rect.top;
    const [wx, wy] = screenToWorld(sx, sy);
    if (insertSelectedVertexAt(wx, wy, !!ev.shiftKey)) {
      ev.preventDefault();
    }
  });
  function hitTestPbbEnd(wx, wy) {
    const maxD2 = (CELL_SIZE * HIT_PBB_END_CF) ** 2;
    const cands = [];
    state.pbbStands.forEach(pbb => {
      const pt = getStandConnectionPx(pbb);
      cands.push({ id: pbb.id, kind: 'pbb', x: pt[0], y: pt[1] });
    });
    state.remoteStands.forEach(st => {
      const [cx, cy] = getRemoteStandCenterPx(st);
      cands.push({ id: st.id, kind: 'remote', x: cx, y: cy });
    });
    const best = findNearestItem(cands, c => [c.x, c.y], wx, wy, maxD2);
    return best || null;
  }

  function hitTestAnyTaxiwayVertex(wx, wy) {
    const click = [wx, wy];
    const maxD2 = (CELL_SIZE * TRY_PBB_MAX_EDGE_CF) ** 2;
    let best = null;
    let bestD2 = maxD2;
    state.taxiways.forEach(tw => {
      if (!tw.vertices || tw.vertices.length < 2) return;
      for (let i = 0; i < tw.vertices.length - 1; i++) {
        const [x1, y1] = cellToPixel(tw.vertices[i].col, tw.vertices[i].row);
        const [x2, y2] = cellToPixel(tw.vertices[i+1].col, tw.vertices[i+1].row);
        const near = closestPointOnSegment([x1, y1], [x2, y2], click);
        if (!near) continue;
        const d2 = dist2(near, click);
        if (d2 < bestD2) {
          bestD2 = d2;
          best = { taxiwayId: tw.id, x: near[0], y: near[1] };
        }
      }
    });
    return best;
  }

  container.addEventListener('mouseup', function(ev) {
    if (ev.button !== 0) return;
    const wasPanning = !!state.isPanning;
    flushDrawNow();
    state.isPanning = false;
    if (state.dragVertex) {
      state.dragVertex = null;
      return;
    }
    if (state.dragTaxiwayVertex) {
      const tw = state.taxiways.find(t => t.id === state.dragTaxiwayVertex.taxiwayId);
      if (tw && typeof syncStartEndFromVertices === 'function') syncStartEndFromVertices(tw);
      state.dragTaxiwayVertex = null;
      if (typeof syncPanelFromState === 'function') syncPanelFromState();
      if (typeof updateObjectInfo === 'function') updateObjectInfo();
      if (typeof redrawLayoutAfterEdit === 'function') redrawLayoutAfterEdit();
      else {
        if (typeof updateAllFlightPaths === 'function') updateAllFlightPaths();
        if (scene3d) update3DScene();
        draw();
      }
      return;
    }
    if (state.dragStandRotation) {
      state.dragStandRotation = null;
      if (typeof syncPanelFromState === 'function') syncPanelFromState();
      if (typeof updateObjectInfo === 'function') updateObjectInfo();
      if (typeof redrawLayoutAfterEdit === 'function') redrawLayoutAfterEdit();
      else {
        if (typeof updateAllFlightPaths === 'function') updateAllFlightPaths();
        if (scene3d) update3DScene();
        draw();
      }
      return;
    }
    if (state.dragPbbBridgeVertex) {
      state.dragPbbBridgeVertex = null;
      updateObjectInfo();
      draw();
      return;
    }
    if (state.dragStandConnection) {
      state.dragStandConnection = null;
      updateObjectInfo();
      if (typeof redrawLayoutAfterEdit === 'function') redrawLayoutAfterEdit();
      else {
        if (typeof updateAllFlightPaths === 'function') updateAllFlightPaths(); else draw();
        if (scene3d) update3DScene();
      }
      return;
    }
    if (state.dragApronLinkVertex) {
      state.dragApronLinkVertex = null;
      if (typeof updateObjectInfo === 'function') updateObjectInfo();
      if (typeof redrawLayoutAfterEdit === 'function') redrawLayoutAfterEdit();
      else {
        if (typeof updateAllFlightPaths === 'function') updateAllFlightPaths();
        if (scene3d) update3DScene();
        draw();
      }
      return;
    }
    const rect = canvas.getBoundingClientRect();
    const sx = ev.clientX - rect.left, sy = ev.clientY - rect.top;
    const [wx, wy] = screenToWorld(sx, sy);
    const placePx = worldPointToPixel(wx, wy, !!ev.shiftKey);
    const mode = settingModeSelect.value;
    const inStandDrawingMode = (mode === 'pbb' && state.pbbDrawing) || (mode === 'remote' && state.remoteDrawing) || (mode === 'holdingPoint' && state.holdingPointDrawing);
    if (!state.dragStart && !inStandDrawingMode) { state.dragStart = null; return; }
    if (handlePbbOrRemoteMouseUp2D(mode, placePx[0], placePx[1])) {
      state.dragStart = null;
      return;
    }
    if (!state.dragStart) return;
    if (!wasPanning) {
      const mode = settingModeSelect.value;
      if (mode === 'edge') {
        rebuildDerivedGraphEdges();
        const eh = hitTestLayoutGraphEdge(wx, wy);
        if (eh) {
          state.selectedObject = { type: 'layoutEdge', id: eh.id, obj: eh };
        } else {
          state.selectedObject = null;
        }
        state.flightPathRevealFlightId = null;
        syncPanelFromState();
        updateObjectInfo();
        draw();
        if (typeof syncAllocGanttSelectionHighlight === 'function') syncAllocGanttSelectionHighlight();
        state.dragStart = null;
        return;
      }
      const hit = hitTest(wx, wy);
      if (mode === 'apronTaxiway' && state.apronLinkDrawing) {
        const pbbHit = hitTestPbbEnd(wx, wy);
        const twHit = hitTestAnyTaxiwayVertex(wx, wy);
        const endpoint = pbbHit ? { kind: pbbHit.kind, standId: pbbHit.id, x: pbbHit.x, y: pbbHit.y } :
                          (twHit ? { kind: 'taxiway', taxiwayId: twHit.taxiwayId, x: twHit.x, y: twHit.y } : null);
        if (endpoint) {
          if (!state.apronLinkTemp) {
            state.apronLinkTemp = endpoint;
            state.apronLinkMidpoints = [];
          } else {
            const first = state.apronLinkTemp;
            if (first.kind !== endpoint.kind) {
              let standId, taxiwayId, tx, ty, midVertices;
              if (first.kind === 'taxiway') {
                taxiwayId = first.taxiwayId;
                standId = endpoint.standId;
                tx = first.x;
                ty = first.y;
                midVertices = (state.apronLinkMidpoints || []).slice().reverse();
              } else {
                taxiwayId = endpoint.taxiwayId;
                standId = first.standId;
                tx = endpoint.x;
                ty = endpoint.y;
                midVertices = (state.apronLinkMidpoints || []).slice();
              }
              if (standId && taxiwayId) {
                const newId = id();
                const inputName = document.getElementById('apronLinkName');
                const linkName = (inputName && String(inputName.value).trim()) || getApronLinkDefaultName(newId);
                if (findDuplicateLayoutName('apronLink', newId, linkName)) {
                  alertDuplicateLayoutName();
                } else {
                  pushUndo();
                  const linkRec = { id: newId, name: linkName, pbbId: standId, taxiwayId, tx, ty };
                  if (midVertices && midVertices.length) linkRec.midVertices = midVertices;
                  state.apronLinks.push(linkRec);
                  syncPanelFromState();
                  if (typeof redrawLayoutAfterEdit === 'function') redrawLayoutAfterEdit();
                  else {
                    if (typeof updateAllFlightPaths === 'function') updateAllFlightPaths();
                    if (scene3d) update3DScene();
                  }
                }
              }
            }
            state.apronLinkTemp = null;
            state.apronLinkMidpoints = [];
            state.apronLinkPointerWorld = null;
          }
          draw();
        } else if (state.apronLinkTemp) {
          const [col, row] = pixelToCell(wx, wy);
          if (col >= 0 && row >= 0 && col <= GRID_COLS && row <= GRID_ROWS) {
            const last = state.apronLinkMidpoints[state.apronLinkMidpoints.length - 1];
            if (!last || last.col !== col || last.row !== row) {
              state.apronLinkMidpoints.push({ col, row });
            }
          }
          draw();
        }
      } else if (hit) {
        state.flightPathRevealFlightId = null;
        state.selectedObject = hit;
        if (hit.type === 'terminal') state.currentTerminalId = hit.id;
        const sm = settingModeValueForHit(hit);
        if (sm) settingModeSelect.value = sm;
        if (hit.type === 'flight' && typeof switchToTab === 'function') switchToTab('flight');
        if (typeof syncSettingsPaneToMode === 'function') syncSettingsPaneToMode();
        syncPanelFromState();
        renderObjectList();
        updateObjectInfo();
        draw();
        if (typeof syncAllocGanttSelectionHighlight === 'function') syncAllocGanttSelectionHighlight();
      } else {
        const pt = worldPointToCellPoint(wx, wy, !!ev.shiftKey);
        const col = pt.col, row = pt.row;
        if (col < 0 || row < 0 || col > GRID_COLS || row > GRID_ROWS) { state.dragStart = null; return; }
        if (mode === 'terminal') {
          if (state.terminalDrawingId) {
            let term = state.terminals.find(t => t.id === state.terminalDrawingId);
            if (!term) {
              state.terminalDrawingId = null;
              state.layoutPathDrawPointer = null;
            } else {
              const pt = { col, row };
              if (term.vertices.length === 0) {
                pushUndo();
                term.vertices.push(pt);
              } else {
                const [fx,fy] = cellToPixel(term.vertices[0].col, term.vertices[0].row);
                const d2 = dist2([fx,fy], cellToPixel(col, row));
                if (d2 < (CELL_SIZE * TERM_CLOSE_POLY_CF) ** 2 && term.vertices.length >= 3) {
                  term.closed = true;
                  state.terminalDrawingId = null;
                  state.layoutPathDrawPointer = null;
                  syncPanelFromState();
                } else {
                  const last = term.vertices[term.vertices.length-1];
                  if (last.col !== col || last.row !== row) { pushUndo(); term.vertices.push(pt); }
                }
              }
              draw();
            }
          }
        } else if (isPathLayoutMode(mode)) {
          if (state.taxiwayDrawingId) {
            const tw = state.taxiways.find(t => t.id === state.taxiwayDrawingId);
            if (tw) {
              const pt = { col, row };
              const last = tw.vertices[tw.vertices.length - 1];
              if (!last || last.col !== col || last.row !== row) {
                if (tw.pathType === 'runway' && tw.vertices.length >= 2) return;
                pushUndo();
                tw.vertices.push(pt);
                if (typeof syncStartEndFromVertices === 'function') syncStartEndFromVertices(tw);
                if (tw.pathType === 'runway' && tw.vertices.length >= 2) {
                  state.taxiwayDrawingId = null;
                  state.layoutPathDrawPointer = null;
                  syncPanelFromState();
                }
                if (typeof redrawLayoutAfterEdit === 'function') redrawLayoutAfterEdit();
                else if (typeof updateAllFlightPaths === 'function') updateAllFlightPaths(); else draw();
              }
            }
          }
        } else if (mode === 'pbb') {
          if (tryPlacePbbAt(wx, wy)) {
            syncPanelFromState();
            draw();
          }
        } else if (mode === 'remote' && state.remoteDrawing) {
          const prev = state.previewRemote;
          if (prev && !prev.overlap && tryPlaceRemoteAt(prev.x, prev.y)) {
            syncPanelFromState();
            draw();
          }
        }
      }
    }
    state.dragStart = null;
  });
  let scene3d = null, camera3d = null, renderer3d = null, controls3d = null, grid3DMapper = null, raycaster3d = null, mouse3d = null, groundPlane3d = null, gridGroup3d = null;
  let mouse3dDown = null;
  const view3dContainer = document.getElementById('view3d-container');
  document.getElementById('btnView2D').classList.add('active');
  document.getElementById('btnView2D').addEventListener('click', function() {
    document.getElementById('btnView2D').classList.add('active');
    document.getElementById('btnView3D').classList.remove('active');
    document.getElementById('canvas-container').style.display = 'block';
    view3dContainer.classList.remove('active');
    if (renderer3d) renderer3d.domElement.style.display = 'none';
    requestAnimationFrame(function() {
      if (typeof resizeCanvas === 'function') resizeCanvas();
    });
  });
  document.getElementById('btnView3D').addEventListener('click', function() {
    document.getElementById('btnView3D').classList.add('active');
    document.getElementById('btnView2D').classList.remove('active');
    document.getElementById('canvas-container').style.display = 'none';
    view3dContainer.classList.add('active');
    init3D();
    animate3D();
  });

  function reset3DView() {
    if (!camera3d) return;
    const halfW = (GRID_COLS * CELL_SIZE) / 2;
    const halfH = (GRID_ROWS * CELL_SIZE) / 2;
    const maxDim = Math.max(halfW, halfH);
    camera3d.position.set(maxDim * 1.2, maxDim * 1.2, maxDim * 1.2);
    camera3d.lookAt(0, 0, 0);
    if (controls3d) {
      controls3d.target.set(0, 0, 0);
      controls3d.update();
    }
  }

  if (resetViewBtn) {
    resetViewBtn.addEventListener('click', function() {
      try {
        resizeCanvas();
        if (view3dContainer.classList.contains('active')) reset3DView();
        else reset2DView();
        try { draw(); } catch(e) {}
        if (typeof update3DScene === 'function') update3DScene();
      } catch (e) { console.error('Fit button error:', e); }
    });
  }
  if (gridToggleBtn) {
    syncGridToggleButton();
    gridToggleBtn.addEventListener('click', function() {
      state.showGrid = !state.showGrid;
      syncGridToggleButton();
      draw();
    });
  }
  if (imageToggleBtn) {
    syncImageToggleButton();
    imageToggleBtn.addEventListener('click', function() {
      state.showImage = !state.showImage;
      syncImageToggleButton();
      invalidateGridUnderlay();
      draw();
    });
  }
  class Grid3DMapper {
    constructor(cols, rows, cellSize) {
      this.cols = cols;
      this.rows = rows;
      this.cellSize = cellSize;
      this.ox = (cols * cellSize) / 2;
      this.oz = (rows * cellSize) / 2;
    }
    pixelToWorldXZ(x, y) {
      return { x: this.ox - x, z: this.oz - y };
    }
    cellToWorld(col, row, height) {
      const [px, py] = cellToPixel(col, row);
      const p = this.pixelToWorldXZ(px, py);
      return new THREE.Vector3(p.x, height, p.z);
    }
    worldFromPixel(x, y, height) {
      const p = this.pixelToWorldXZ(x, y);
      return new THREE.Vector3(p.x, height, p.z);
    }
    shapeFromCell(col, row) {
      const [px, py] = cellToPixel(col, row);
      return { x: this.ox - px, y: py - this.oz };
    }
    worldToPixel(xWorld, zWorld) {
      return { x: this.ox - xWorld, y: this.oz - zWorld };
    }
    worldToCell(xWorld, zWorld) {
      const p = this.worldToPixel(xWorld, zWorld);
      let col = Math.round(p.x / this.cellSize);
      let row = Math.round(p.y / this.cellSize);
      col = Math.max(0, Math.min(this.cols, col));
      row = Math.max(0, Math.min(this.rows, row));
      return [col, row];
    }
  }

  function init3D() {
    if (renderer3d) { renderer3d.domElement.style.display = 'block'; update3DScene(); return; }
    const w = view3dContainer.clientWidth, h = view3dContainer.clientHeight;
    scene3d = new THREE.Scene();
    scene3d.background = new THREE.Color(GRID_VIEW_BG);
    gridGroup3d = new THREE.Group();
    scene3d.add(gridGroup3d);
    camera3d = new THREE.PerspectiveCamera(50, w/h, 1, 100000);
    const halfW = (GRID_COLS * CELL_SIZE) / 2, halfH = (GRID_ROWS * CELL_SIZE) / 2;
    const maxDim = Math.max(halfW, halfH);
    camera3d.position.set(maxDim * 1.2, maxDim * 1.2, maxDim * 1.2);
    camera3d.lookAt(0, 0, 0);
    const axisLen = CELL_SIZE * 8;
    const axisOrigin = new THREE.Vector3(-maxDim, 0, -maxDim);
    function addAxis(toVec, color) {
      const pts = [axisOrigin, axisOrigin.clone().add(toVec)];
      const geo = new THREE.BufferGeometry().setFromPoints(pts);
      const mat = new THREE.LineBasicMaterial({ color });
      const line = new THREE.Line(geo, mat);
      gridGroup3d.add(line);
    }
    addAxis(new THREE.Vector3(axisLen, 0, 0), 0xef4444);
    addAxis(new THREE.Vector3(0, 0, axisLen), 0x22c55e);
    addAxis(new THREE.Vector3(0, axisLen, 0), 0x7c6af7);
    function createAxisLabel(text, color, endVec) {
      const size = 128;
      const canvasLabel = document.createElement('canvas');
      canvasLabel.width = size;
      canvasLabel.height = size;
      const g = canvasLabel.getContext('2d');
      g.clearRect(0, 0, size, size);
      g.font = 'bold 72px system-ui';
      g.fillStyle = color;
      g.textAlign = 'center';
      g.textBaseline = 'middle';
      g.fillText(text, size / 2, size / 2);
      const tex = new THREE.CanvasTexture(canvasLabel);
      const mat = new THREE.SpriteMaterial({ map: tex, transparent: true });
      const sprite = new THREE.Sprite(mat);
      const s = CELL_SIZE * 3;
      sprite.scale.set(s, s, 1);
      sprite.position.copy(axisOrigin.clone().add(endVec));
      gridGroup3d.add(sprite);
    }
    createAxisLabel('x', '#ef4444', new THREE.Vector3(axisLen * 1.1, 0, 0));
    createAxisLabel('y', '#22c55e', new THREE.Vector3(0, 0, axisLen * 1.1));
    createAxisLabel('z', '#7c6af7', new THREE.Vector3(0, axisLen * 1.1, 0));
    grid3DMapper = new Grid3DMapper(GRID_COLS, GRID_ROWS, CELL_SIZE);
    renderer3d = new THREE.WebGLRenderer({ antialias: true });
    renderer3d.setSize(w, h);
    renderer3d.setPixelRatio(window.devicePixelRatio || 1);
    view3dContainer.appendChild(renderer3d.domElement);
    controls3d = new THREE.OrbitControls(camera3d, renderer3d.domElement);
    controls3d.enableDamping = true;
    controls3d.dampingFactor = 0.05;
    raycaster3d = new THREE.Raycaster();
    mouse3d = new THREE.Vector2();
    groundPlane3d = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
    const dom3d = renderer3d.domElement;
    function getHitPoint(ev) {
      const rect = dom3d.getBoundingClientRect();
      const ndcX = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
      const ndcY = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
      mouse3d.set(ndcX, ndcY);
      raycaster3d.setFromCamera(mouse3d, camera3d);
      const hit = new THREE.Vector3();
      return raycaster3d.ray.intersectPlane(groundPlane3d, hit) ? hit : null;
    }
    dom3d.addEventListener('mousedown', function(ev) {
      if (ev.button === 0) mouse3dDown = { x: ev.clientX, y: ev.clientY };
    });
    dom3d.addEventListener('mouseup', function(ev) {
      if (ev.button !== 0 || !mouse3dDown) return;
      const dx = ev.clientX - mouse3dDown.x, dy = ev.clientY - mouse3dDown.y;
      if (dx*dx + dy*dy > 25) { mouse3dDown = null; return; }
      mouse3dDown = null;
      const hit = getHitPoint(ev);
      if (!hit || !grid3DMapper) return;
      const mode = settingModeSelect.value;
      const p = grid3DMapper.worldToPixel(hit.x, hit.z);
      const wx = p.x, wy = p.y;
      const [col, row] = grid3DMapper.worldToCell(hit.x, hit.z);
      tryCommitStandPlacement3D(mode, wx, wy, col, row);
    });
    const step = CELL_SIZE;
    const faintLines = [];
    const majorLines = [];
    let kx = 0;
    for (let x = -maxDim; x <= maxDim; x += step, kx++) {
      const pts = [new THREE.Vector3(x, 0, -maxDim), new THREE.Vector3(x, 0, maxDim)];
      if (kx % GRID_MAJOR_INTERVAL === 0) majorLines.push.apply(majorLines, pts);
      else faintLines.push.apply(faintLines, pts);
    }
    let kz = 0;
    for (let z = -maxDim; z <= maxDim; z += step, kz++) {
      const pts = [new THREE.Vector3(-maxDim, 0, z), new THREE.Vector3(maxDim, 0, z)];
      if (kz % GRID_MAJOR_INTERVAL === 0) majorLines.push.apply(majorLines, pts);
      else faintLines.push.apply(faintLines, pts);
    }
    if (faintLines.length) {
      const faintGeo = new THREE.BufferGeometry().setFromPoints(faintLines);
      const faintMat = new THREE.LineBasicMaterial({
        color: 0xd4d4d4,
        transparent: true,
        opacity: 0.2,
        depthTest: false
      });
      gridGroup3d.add(new THREE.LineSegments(faintGeo, faintMat));
    }
    if (majorLines.length) {
      const majorGeo = new THREE.BufferGeometry().setFromPoints(majorLines);
      const majorMat = new THREE.LineBasicMaterial({
        color: 0xffffff,
        transparent: true,
        opacity: 0.35,
        depthTest: false
      });
      gridGroup3d.add(new THREE.LineSegments(majorGeo, majorMat));
    }
    update3DScene();
  }

  function update3DScene() {
    if (!scene3d) return;
    while (scene3d.children.length > 1) scene3d.remove(scene3d.children[scene3d.children.length - 1]);
    if (!grid3DMapper) grid3DMapper = new Grid3DMapper(GRID_COLS, GRID_ROWS, CELL_SIZE);
  }

  function animate3D() {
    if (!renderer3d || !view3dContainer.classList.contains('active')) return;
    requestAnimationFrame(animate3D);
    if (controls3d) controls3d.update();
    if (renderer3d && scene3d && camera3d) renderer3d.render(scene3d, camera3d);
  }

  container.addEventListener('wheel', function(ev) {
    ev.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const mx = ev.clientX - rect.left, my = ev.clientY - rect.top;
    const wx = (mx - state.panX) / state.scale, wy = (my - state.panY) / state.scale;
    const factor = 1 - ev.deltaY * 0.002;
    state.scale *= factor;
    state.scale = Math.max(CANVAS_MIN_ZOOM, Math.min(CANVAS_MAX_ZOOM, state.scale));
    state.panX = mx - wx * state.scale;
    state.panY = my - wy * state.scale;
    try { draw(); } catch(e) {}
  }, { passive: false });

  window.addEventListener('resize', function() {
    resizeCanvas();
    if (renderer3d && view3dContainer.classList.contains('active')) {
      const w = view3dContainer.clientWidth, h = view3dContainer.clientHeight;
      camera3d.aspect = w / h;
      camera3d.updateProjectionMatrix();
      renderer3d.setSize(w, h);
    }
  });
  try { applyInitialLayoutFromJson(); } catch(applyErr) { console.error('Layout apply failed:', applyErr); }
  updateLayoutNameBar(INITIAL_LAYOUT_DISPLAY_NAME || 'default_layout');
  resizeCanvas();
  reset2DView();
  syncPanelFromState();
  if (typeof draw === 'function') draw();
  if (typeof update3DScene === 'function') update3DScene();
  if (typeof renderKpiDashboard === 'function') renderKpiDashboard('Initial load');
})();
