    const input = document.getElementById('markerTextDraftInput');
    if (!layer || !input) return;
    layer.removeAttribute('hidden');
    layer.setAttribute('aria-hidden', 'false');
    input.value = '';
    syncMarkerTextDraftInputPosition();
    setTimeout(function() {
      try {
        input.focus();
      } catch (e) {}
    }, 0);
  }
  function commitMarkerTextDraft() {
    const d = state.markerTextDraft;
    if (!d || !d.active) return;
    const input = document.getElementById('markerTextDraftInput');
    const text = input ? String(input.value || '').trim().slice(0, 500) : '';
    const sx = d.x, sy = d.y;
    state.markerTextDraft = null;
    hideMarkerTextDraftEditor();
    if (text) {
      pushUndo();
      state.layoutMarkers.push({ kind: 'text', id: id(), x: sx, y: sy, text: text });
      syncPanelFromState();
    }
    scheduleDraw();
  }
  function cancelMarkerTextDraftWithoutCommit() {
    if (!state.markerTextDraft || !state.markerTextDraft.active) return;
    state.markerTextDraft = null;
    hideMarkerTextDraftEditor();
    scheduleDraw();
  }
  function handleMarkerPlacement(wx, wy, shiftKey) {
    const placePx = worldPointToPixel(wx, wy, shiftKey);
    const sub = getMarkerSubKindFromPanel();
    const placeUse = sub === 'area' ? markerAreaSnapWorldToPlacementPx(wx, wy, shiftKey) : placePx;
    const px = placeUse[0], py = placeUse[1];
    if (sub !== 'text' && state.markerTextDraft && state.markerTextDraft.active) {
      commitMarkerTextDraft();
    }
    if (sub === 'text') {
      commitMarkerTextDraft();
      state.markerTextDraft = { x: px, y: py, active: true };
      showMarkerTextDraftEditor();
      scheduleDraw();
      return;
    }
    if (sub === 'ruler') {
      if (!state.markerRulerDraft) {
        state.markerRulerDraft = { x: px, y: py };
        state.markerRulerHoverWorld = [px, py];
      } else {
        const x1 = state.markerRulerDraft.x, y1 = state.markerRulerDraft.y;
        state.markerRulerDraft = null;
        state.markerRulerHoverWorld = null;
        const dx = px - x1, dy = py - y1;
        if (dx * dx + dy * dy < 2.25) return;
        pushUndo();
        state.layoutMarkers.push({ kind: 'ruler', id: id(), x1: x1, y1: y1, x2: px, y2: py });
        syncPanelFromState();
      }
      return;
    }
    if (sub === 'island') {
      if (!state.markerIslandDraft) state.markerIslandDraft = { points: [] };
      const draft = state.markerIslandDraft;
      const list = draft.points;
      const closeR = CELL_SIZE * TERM_CLOSE_POLY_CF;
      const closeR2 = closeR * closeR;
      if (list.length >= 3) {
        const c0 = list[0];
        const dx = px - c0.x, dy = py - c0.y;
        if (dx * dx + dy * dy <= closeR2) {
          pushUndo();
          state.layoutMarkers.push({
            kind: 'island',
            id: id(),
            points: list.map(function(p) { return { x: Number(p.x), y: Number(p.y) }; }),
            widthM: getMarkerIslandWidthMFromPanel()
          });
          state.markerIslandDraft = null;
          state.markerIslandHoverWorld = null;
          syncPanelFromState();
          return;
        }
      }
      list.push({ x: px, y: py });
      state.markerIslandHoverWorld = [px, py];
      return;
    }
    if (sub === 'area') {
      if (!state.markerAreaDraft) state.markerAreaDraft = { points: [] };
      const draftA = state.markerAreaDraft;
      const listA = draftA.points;
      const closeRa = CELL_SIZE * TERM_CLOSE_POLY_CF;
      const closeRa2 = closeRa * closeRa;
      if (listA.length >= 3) {
        const c0a = listA[0];
        const dxa = px - c0a.x, dya = py - c0a.y;
        if (dxa * dxa + dya * dya <= closeRa2) {
          pushUndo();
          state.layoutMarkers = normalizeLayoutMarkerAreaZOrder((state.layoutMarkers || []).concat([{
            kind: 'area',
            id: id(),
            points: listA.map(function(p) { return { x: Number(p.x), y: Number(p.y) }; }),
          }]));
          state.markerAreaDraft = null;
          state.markerAreaHoverWorld = null;
          syncPanelFromState();
          return;
        }
      }
      listA.push({ x: px, y: py });
      state.markerAreaHoverWorld = [px, py];
      return;
    }
    if (sub === 'flight') {
      const snap = snapWorldToMarkerFlightTaxiway(wx, wy);
      if (!snap) return;
      pushUndo();
      state.layoutMarkers.push({
        kind: 'flight',
        id: id(),
        taxiwayId: snap.taxiwayId,
        segIndex: snap.segIndex,
        t: snap.t,
        aircraftType: getMarkerFlightAircraftTypeFromPanel(),
        blazerEnabled: false,
        headingReversed: false,
        blazerColor: MARKER_BLAZER_COLOR_OPTIONS[0],
        blazerLeftTrail: [],
        blazerRightTrail: [],
      });
      syncPanelFromState();
      return;
    }
    if (sub === 'navaid') {
      pushUndo();
      state.layoutMarkers.push({
        kind: 'navaid',
        id: id(),
        subType: getMarkerNavaidTypeFromPanel(),
        x: px,
        y: py,
      });
      syncPanelFromState();
      return;
    }
  }

  function hitTest(wx, wy) {
    const click = [wx, wy];
    if (layoutMarkersVisible()) {
      // Marker flights are drawn on top; prioritize marker picking so clicks don't fall through.
      const mkTopHit = hitTestLayoutMarker(wx, wy, { skipKind: 'area' });
      if (mkTopHit) return mkTopHit;
    }
    const temps = state.tempStands || [];
    for (let i = temps.length - 1; i >= 0; i--) {
      const st = temps[i];
      if (pointInPolygonXY([wx, wy], getRemoteStandCorners(st)))
        return { type: 'tempStand', id: st.id, obj: st };
    }
    for (let i = state.remoteStands.length - 1; i >= 0; i--) {
      const st = state.remoteStands[i];
      if (pointInPolygonXY([wx, wy], getRemoteStandCorners(st)))
        return { type: 'remote', id: st.id, obj: st };
    }
    for (let i = state.pbbStands.length - 1; i >= 0; i--) {
      const pbb = state.pbbStands[i];
      const corners = getPBBStandCorners(pbb);
      if (pointInPolygonXY(click, corners))
        return { type: 'pbb', id: pbb.id, obj: pbb };
    }
    for (let i = state.terminals.length - 1; i >= 0; i--) {
      const t = state.terminals[i];
      if (t.closed && t.vertices.length >= 3 && pointInPolygon(click, t.vertices))
        return { type: 'terminal', id: t.id, obj: t };
    }
    const hpHit = hitTestHoldingPoint(wx, wy);
    if (hpHit) return hpHit;
    const apronLkHit = hitTestApronLink(wx, wy);
    if (apronLkHit) return apronLkHit;
    if (!state.taxiwayDrawingId) {
      const pathCenterlineHitRadiusWorld = 10 / Math.max(state.scale, 0.08);
      const pathCenterlineHitD2 = pathCenterlineHitRadiusWorld * pathCenterlineHitRadiusWorld;
      for (let i = state.taxiways.length - 1; i >= 0; i--) {
        const tw = state.taxiways[i];
        if (tw.vertices.length < 2) continue;
        for (let j = 0; j < tw.vertices.length - 1; j++) {
          const [x1, y1] = cellToPixel(tw.vertices[j].col, tw.vertices[j].row);
          const [x2, y2] = cellToPixel(tw.vertices[j + 1].col, tw.vertices[j + 1].row);
          const near = closestPointOnSegment([x1, y1], [x2, y2], click);
          if (near && dist2(near, click) < pathCenterlineHitD2) return { type: 'taxiway', id: tw.id, obj: tw };
        }
      }
    }
    if (layoutMarkersVisible()) {
      const arHit = hitTestLayoutMarker(wx, wy, { onlyKind: 'area' });
      if (arHit) return arHit;
    }
    return null;
  }

  function hitTestSimFlightAtWorld(wx, wy) {
    if (!simPlaybackVisualsActive() || simPlaybackHeavyVisualsSuppressed() || !state.flights || !state.flights.length) return null;
    const tSec = state.simTimeSec;
    let best = null;
    let bestD2 = Infinity;
    const flights = state.flights;
    for (let i = 0; i < flights.length; i++) {
      const f = flights[i];
      if (!f || flightBlockedLikeNoWay(f)) continue;
      const pose = getFlightPoseAtTimeForDraw(f, tSec);
      if (!pose) continue;
      const dx = pose.x - wx, dy = pose.y - wy;
      const d2 = dx * dx + dy * dy;
      const poly = simFlightSilhouetteWorldPolygon(f, pose, tSec);
      if (poly.length >= 3 && pointInPolygonXY([wx, wy], poly) && d2 < bestD2) {
        bestD2 = d2;
        best = f;
      }
    }
    return best;
  }

  function hitTestTerminalVertex(wx, wy) {
    const maxD2 = (CELL_SIZE * HIT_TERM_VTX_CF) ** 2;
    const cands = [];
    state.terminals.forEach(t => {
      t.vertices.forEach((v, idx) => {
        cands.push({ terminalId: t.id, index: idx, v });
      });
    });
    const best = findNearestItem(cands, c => cellToPixel(c.v.col, c.v.row), wx, wy, maxD2);
    return best ? { terminalId: best.terminalId, index: best.index } : null;
  }

  function hitTestTaxiwayVertex(wx, wy) {
    if (!state.selectedObject || state.selectedObject.type !== 'taxiway') return null;
    const tw = state.selectedObject.obj;
    if (!tw || !tw.vertices || tw.vertices.length === 0) return null;
    const click = [wx, wy];
    const maxD2 = (CELL_SIZE * HIT_TW_VTX_CF) ** 2;
    let best = null;
    let bestD2 = maxD2;
    tw.vertices.forEach((v, idx) => {
      const [vx, vy] = cellToPixel(v.col, v.row);
      const d2 = dist2([vx, vy], click);
      if (d2 < bestD2) {
        bestD2 = d2;
        best = { taxiwayId: tw.id, index: idx };
      }
    });
    return best;
  }
  function hitTestPbbEditablePoint(wx, wy) {
    if (!state.selectedObject || state.selectedObject.type !== 'pbb') return null;
    const pbb = state.selectedObject.obj;
    if (!pbb || pbb.id !== state.selectedObject.id) return null;
    const click = [wx, wy];
    const maxD2 = (CELL_SIZE * HIT_PBB_END_CF) ** 2;
    let best = null;
    let bestD2 = maxD2;
    (Array.isArray(pbb.pbbBridges) ? pbb.pbbBridges : []).forEach(function(bridge, bridgeIdx) {
      (Array.isArray(bridge.points) ? bridge.points : []).forEach(function(pt, ptIdx) {
        if (ptIdx === 1) return;
        const d2 = dist2([Number(pt.x) || 0, Number(pt.y) || 0], click);
        if (d2 < bestD2) {
          bestD2 = d2;
          best = { type: 'bridge', bridgeIndex: bridgeIdx, pointIndex: ptIdx };
        }
      });
    });
    const apronPt = getStandAircraftMarkerWorldPxForPbb(pbb);
    const apronD2 = dist2(apronPt, click);
    if (apronD2 < bestD2) best = { type: 'apronSite' };
    return best;
  }
  function hitTestRemoteStandDragPoint(wx, wy) {
    if (!state.selectedObject || state.selectedObject.type !== 'remote') return null;
    const st = state.selectedObject.obj;
    if (!st || st.id !== state.selectedObject.id) return null;
    const click = [wx, wy];
    const maxD2 = (CELL_SIZE * HIT_PBB_END_CF) ** 2;
    const mk = getStandAircraftMarkerWorldPxForRemoteLike(st);
    if (dist2(mk, click) <= maxD2) return { type: 'remoteCenter' };
    return null;
  }
  function findInsertSegment(vertices, closed, wx, wy) {
    if (!Array.isArray(vertices) || vertices.length < 2) return null;
    const click = [wx, wy];
    const maxD2 = (CELL_SIZE * INSERT_VERTEX_HIT_CF) ** 2;
    let best = null;
    let bestD2 = maxD2;
    const lastSeg = closed ? vertices.length : (vertices.length - 1);
    function vertexToPixel(v) {
      if (Array.isArray(v) && v.length >= 2) return [Number(v[0]) || 0, Number(v[1]) || 0];
      if (v && v.x != null && v.y != null) return [Number(v.x) || 0, Number(v.y) || 0];
      return cellToPixel(v.col, v.row);
    }
    for (let i = 0; i < lastSeg; i++) {
      const curr = vertices[i];
      const next = vertices[(i + 1) % vertices.length];
      const p1 = vertexToPixel(curr);
      const p2 = vertexToPixel(next);
      const near = closestPointOnSegment(p1, p2, click);
      if (!near) continue;
      const d2 = dist2(near, click);
      if (d2 < bestD2) {
        bestD2 = d2;
        best = { insertIndex: i + 1, near: near };
      }
    }
    return best;
  }
  const PATH_ARC_MIN_BULGE_PX = 2;
  const PATH_ARC_MAX_BULGE_FRAC = 0.45;
  function pathArcAngleDiffCCW(t0, t1) {
    let d = t1 - t0;
    while (d < 0) d += 2 * Math.PI;
    while (d >= 2 * Math.PI) d -= 2 * Math.PI;
    return d;
  }
  function pathArcPointBetweenAnglesCCW(tStart, tProbe, spanCCW) {
    return pathArcAngleDiffCCW(tStart, tProbe) <= spanCCW + 1e-10;
  }
  function pathArcCircumcircle(ax, ay, bx, by, cx, cy) {
    const d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
    if (Math.abs(d) < 1e-12) return null;
    const a2 = ax * ax + ay * ay;
    const b2 = bx * bx + by * by;
    const c2 = cx * cx + cy * cy;
    const ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d;
    const uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d;
    const r = Math.hypot(ax - ux, ay - uy);
    if (!(r > 1e-9)) return null;
    return { ox: ux, oy: uy, r: r };
  }
  /** Endpoints A,B and point C on arc; returns world px polyline A→B along the circle through C. */
  function pathArcSampleThreePointWorldPx(ax, ay, bx, by, cx, cy, maxChordStepPx) {
    const cc = pathArcCircumcircle(ax, ay, bx, by, cx, cy);
    if (!cc) return [[ax, ay], [bx, by]];
    const ta = Math.atan2(ay - cc.oy, ax - cc.ox);
    const tb = Math.atan2(by - cc.oy, bx - cc.ox);
    const tc = Math.atan2(cy - cc.oy, cx - cc.ox);
    const spanAB = pathArcAngleDiffCCW(ta, tb);
    let tStart, span, reverseOrder;
    if (pathArcPointBetweenAnglesCCW(ta, tc, spanAB)) {
      tStart = ta;
      span = spanAB;
      reverseOrder = false;
    } else {
      tStart = tb;
      span = pathArcAngleDiffCCW(tb, ta);
      reverseOrder = true;
    }
    const arcLen = cc.r * span;
    const mcs = Math.max(3, typeof maxChordStepPx === 'number' && maxChordStepPx > 0 ? maxChordStepPx : CELL_SIZE * 0.28);
    const n = Math.max(8, Math.min(96, Math.ceil(arcLen / mcs)));
    const pts = [];
    for (let i = 0; i <= n; i++) {
      const ang = tStart + (span * i) / n;
      pts.push([cc.ox + cc.r * Math.cos(ang), cc.oy + cc.r * Math.sin(ang)]);
    }
    if (reverseOrder) pts.reverse();
    pts[0] = [ax, ay];
    pts[pts.length - 1] = [bx, by];
    return pts;
  }
  /** Subdivide polyline so each segment length ≤ maxStepPx (smoother grid snap for arcs). */
  function pathArcDensifyPolylinePx(pts, maxStepPx) {
    if (!pts || pts.length < 2) return pts ? pts.slice() : [];
    const m = Math.max(1e-6, maxStepPx);
    const out = [[pts[0][0], pts[0][1]]];
    for (let i = 0; i < pts.length - 1; i++) {
      const x0 = pts[i][0], y0 = pts[i][1], x1 = pts[i + 1][0], y1 = pts[i + 1][1];
      const len = Math.hypot(x1 - x0, y1 - y0);
      const steps = Math.max(1, Math.ceil(len / m));
      for (let s = 1; s <= steps; s++) {
        const t = s / steps;
        out.push([x0 + (x1 - x0) * t, y0 + (y1 - y0) * t]);
      }
    }
