      ctx.font = '600 ' + fs + 'px system-ui,sans-serif';
      tw = Math.max(tw, ctx.measureText(txt).width + 8);
      ctx.restore();
    }
    const th = fs + 8;
    return { left: hx + 2, top: hy + 2, w: tw, h: th };
  }
  function hitTestLayoutMarker(wx, wy, opts) {
    if (!layoutMarkersVisible()) return null;
    const onlyKind = opts && opts.onlyKind;
    const skipKind = opts && opts.skipKind;
    const click = [wx, wy];
    const padW = 6 / Math.max(state.scale, 0.1);
    const list = state.layoutMarkers || [];
    for (let i = list.length - 1; i >= 0; i--) {
      const m = list[i];
      if (!m) continue;
      if (onlyKind && m.kind !== onlyKind) continue;
      if (skipKind && m.kind === skipKind) continue;
      if (m.kind === 'text') {
        const ax = Number(m.x), ay = Number(m.y);
        if (isFinite(ax) && isFinite(ay)) {
          const ar = layoutMarkerHandleHitRadiusWorld() * 1.15;
          if (dist2(click, [ax, ay]) <= ar * ar)
            return { type: 'layoutMarker', id: m.id, obj: m };
        }
        const r = layoutMarkerTextHitRect(m);
        if (!r) continue;
        if (click[0] >= r.left - padW && click[0] <= r.left + r.w + padW &&
            click[1] >= r.top - padW && click[1] <= r.top + r.h + padW)
          return { type: 'layoutMarker', id: m.id, obj: m };
      } else if (m.kind === 'ruler') {
        const x1 = Number(m.x1), y1 = Number(m.y1), x2 = Number(m.x2), y2 = Number(m.y2);
        if (![x1, y1, x2, y2].every(isFinite)) continue;
        const er = layoutMarkerHandleHitRadiusWorld() * 1.1;
        const er2 = er * er;
        if (dist2(click, [x1, y1]) <= er2 || dist2(click, [x2, y2]) <= er2)
          return { type: 'layoutMarker', id: m.id, obj: m };
        const pr = projectOnSegment([x1, y1], [x2, y2], click);
        if (pr.t < 0 || pr.t > 1) continue;
        const tol = Math.max(CELL_SIZE * 0.35, 10 / Math.max(state.scale, 0.12));
        if (dist2(pr.p, click) <= tol * tol)
          return { type: 'layoutMarker', id: m.id, obj: m };
      } else if (m.kind === 'flight') {
        const pose = resolveMarkerFlightPose(m);
        if (!pose) continue;
        const tol = Math.max(CELL_SIZE * 1.1, 22 / Math.max(state.scale, 0.12));
        if (dist2(click, [pose.x, pose.y]) <= tol * tol)
          return { type: 'layoutMarker', id: m.id, obj: m };
      } else if (m.kind === 'island' || m.kind === 'area') {
        const pts = m.points;
        if (!pts || pts.length < 3) continue;
        const poly = pts.map(function(p) { return [Number(p.x), Number(p.y)]; }).filter(function(P) { return isFinite(P[0]) && isFinite(P[1]); });
        if (poly.length < 3) continue;
        const contourLineOnly = m.kind === 'island' && m.id != null && String(m.id).indexOf('contour-') === 0;
        const er = layoutMarkerHandleHitRadiusWorld() * 1.1;
        const er2 = er * er;
        let nearVertex = false;
        for (let ii = 0; ii < poly.length; ii++) {
          if (dist2(click, poly[ii]) <= er2) {
            nearVertex = true;
            break;
          }
        }
        if (nearVertex) return { type: 'layoutMarker', id: m.id, obj: m };
        if (!contourLineOnly && pointInPolygonXY(click, poly)) return { type: 'layoutMarker', id: m.id, obj: m };
        const tol = Math.max(CELL_SIZE * 0.35, 10 / Math.max(state.scale, 0.12));
        const tol2 = tol * tol;
        const nn = poly.length;
        for (let ei = 0; ei < nn; ei++) {
          const p0 = poly[ei], p1 = poly[(ei + 1) % nn];
          const pr = projectOnSegment(p0, p1, click);
          if (pr.t < 0 || pr.t > 1) continue;
          if (dist2(pr.p, click) <= tol2) return { type: 'layoutMarker', id: m.id, obj: m };
        }
      } else if (m.kind === 'navaid') {
        const ax = Number(m.x), ay = Number(m.y);
        if (!isFinite(ax) || !isFinite(ay)) continue;
        const tol = Math.max(CELL_SIZE * 0.8, 18 / Math.max(state.scale, 0.12));
        const tol2 = tol * tol;
        const sub = (m.subType === 'ils') ? 'ils' : 'papi';
        if (sub === 'papi') {
          const xs = papiLampCenterXsWorld(ax);
          let hitP = false;
          for (let pi = 0; pi < 4; pi++) {
            if (dist2(click, [xs[pi], ay]) <= tol2) { hitP = true; break; }
          }
          if (!hitP && dist2(click, [ax, ay]) <= tol2) hitP = true;
          if (!hitP) {
            const half = 1.5 * PAPI_LAMP_SPACING_WORLD + tol;
            if (Math.abs(click[1] - ay) <= tol && click[0] >= ax - half && click[0] <= ax + half) hitP = true;
          }
          if (hitP) return { type: 'layoutMarker', id: m.id, obj: m };
        } else if (dist2(click, [ax, ay]) <= tol2) {
          return { type: 'layoutMarker', id: m.id, obj: m };
        }
      }
    }
    return null;
  }
  /** Navaid: PAPI as four lamps (2 white, 2 red); ILS dot + ILS label at (m.x, m.y). */
  function drawNavaidMarker2D(ctx2, m, selected, interactiveLite) {
    if (!m) return;
    const x = Number(m.x), y = Number(m.y);
    if (!isFinite(x) || !isFinite(y)) return;
    const sub = (m.subType === 'ils') ? 'ils' : 'papi';
    const isIls = sub === 'ils';
    const scaleRef = Math.max(state.scale, 0.1);
    const etcMono = layerMonoEtcOn() && !selected;
    ctx2.save();
    if (!isIls) {
      const lampXs = papiLampCenterXsWorld(x);
      const rLight = Math.max(2.4 * PAPI_VISUAL_SCALE, 2.9 * PAPI_VISUAL_SCALE / scaleRef);
      const fills = selected
        ? ['#ffffff', '#ffffff', '#fca5a5', '#fca5a5']
        : (etcMono
          ? [C2D_LAYER_MONO_ETC_WHITE, C2D_LAYER_MONO_ETC_WHITE, C2D_LAYER_MONO_ETC_WHITE, C2D_LAYER_MONO_ETC_WHITE]
          : ['#f8fafc', '#f8fafc', '#ef4444', '#ef4444']);
      const strokes = selected
        ? [c2dObjectSelectedStroke(), c2dObjectSelectedStroke(), c2dObjectSelectedStroke(), c2dObjectSelectedStroke()]
        : (etcMono
          ? [c2dLayerMonoLineStrokeCss(), c2dLayerMonoLineStrokeCss(), c2dLayerMonoLineStrokeCss(), c2dLayerMonoLineStrokeCss()]
          : ['rgba(148,163,184,0.95)', 'rgba(148,163,184,0.95)', 'rgba(127,29,29,0.98)', 'rgba(127,29,29,0.98)']);
      for (let i = 0; i < 4; i++) {
        ctx2.beginPath();
        ctx2.arc(lampXs[i], y, rLight, 0, Math.PI * 2);
        ctx2.fillStyle = fills[i];
        ctx2.strokeStyle = strokes[i];
        ctx2.lineWidth = Math.max(0.35, 0.55 / scaleRef);
        ctx2.fill();
        ctx2.stroke();
      }
      if (selected) {
        const pad = 2 * PAPI_VISUAL_SCALE;
        const x0 = lampXs[0] - rLight - pad;
        const x1 = lampXs[3] + rLight + pad;
        const y0 = y - rLight - pad;
        const y1 = y + rLight + pad;
        ctx2.strokeStyle = c2dObjectSelectedStroke();
        ctx2.lineWidth = Math.max(0.55, 0.8 / scaleRef);
        ctx2.setLineDash([4, 3]);
        ctx2.strokeRect(x0, y0, x1 - x0, y1 - y0);
        ctx2.setLineDash([]);
      }
      ctx2.restore();
      return;
    }
    const label = 'ILS';
    const fill = selected ? c2dObjectSelectedFill() : (etcMono ? C2D_LAYER_MONO_ETC_WHITE : 'rgba(56, 189, 248, 0.85)');
    const stroke = selected ? c2dObjectSelectedStroke() : (etcMono ? c2dLayerMonoLineStrokeCss() : 'rgba(2, 132, 199, 0.95)');
    const fg = etcMono ? C2D_LAYER_MONO_ETC_WHITE : '#0c4a6e';
    const r = Math.max(3, 3.6 / scaleRef);
    ctx2.beginPath();
    ctx2.arc(x, y, r, 0, Math.PI * 2);
    ctx2.fillStyle = fill;
    ctx2.strokeStyle = stroke;
    ctx2.lineWidth = Math.max(0.4, 0.6 / scaleRef);
    ctx2.fill();
    ctx2.stroke();
    if (!interactiveLite) {
      const fs = Math.max(9, 10 / Math.max(state.scale, 0.12));
      ctx2.font = '700 ' + fs + 'px system-ui,sans-serif';
      ctx2.textAlign = 'left';
      ctx2.textBaseline = 'middle';
      ctx2.lineWidth = 2.4;
      ctx2.strokeStyle = 'rgba(15,23,42,0.85)';
      ctx2.fillStyle = fg;
      const lx = x + r + 3;
      const ly = y;
      ctx2.strokeText(label, lx, ly);
      ctx2.fillText(label, lx, ly);
    }
    ctx2.restore();
  }
  function hideMarkerTextDraftEditor() {
    const layer = document.getElementById('marker-text-edit-layer');
    const input = document.getElementById('markerTextDraftInput');
    if (layer) {
      layer.setAttribute('hidden', '');
      layer.setAttribute('aria-hidden', 'true');
    }
    if (input) input.value = '';
  }
  function syncMarkerTextDraftInputPosition() {
    const draft = state.markerTextDraft;
    const input = document.getElementById('markerTextDraftInput');
    if (!draft || !draft.active || !input) return;
    const sc = worldToScreenCanvas(draft.x, draft.y);
    input.style.left = Math.round(sc[0] + 4) + 'px';
    input.style.top = Math.round(sc[1] + 4) + 'px';
  }
  function showMarkerTextDraftEditor() {
    const layer = document.getElementById('marker-text-edit-layer');
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
