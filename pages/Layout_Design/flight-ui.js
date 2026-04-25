      if (trail.length > 4000) trail.splice(0, trail.length - 4000);
    }
    append(m.blazerLeftTrail, tips.left);
    append(m.blazerRightTrail, tips.right);
  }
  function markerFlightBoundsWorld(m) {
    if (!m || m.kind !== 'flight') return null;
    const pose = resolveMarkerFlightPose(m);
    if (!pose) return null;
    const ac = getAircraftInfoByType(m.aircraftType);
    const lenM = ac && isFinite(Number(ac.length_m)) ? Math.max(1, Number(ac.length_m)) : 40;
    const spanM = ac && isFinite(Number(ac.wingspan_m)) ? Math.max(1, Number(ac.wingspan_m)) : 40;
    const sil = getApronAircraftDetailedSilhouettePoints();
    const localPts = [];
    if (_ac2d.useDetailedSilhouette === true && sil.length >= 3) {
      for (let i = 0; i < sil.length; i++) {
        const p = sil[i];
        if (!p || p.length < 2) continue;
        const lx = Number(p[0]) * lenM;
        const ly = Number(p[1]) * spanM;
        if (isFinite(lx) && isFinite(ly)) localPts.push([lx, ly]);
      }
    }
    if (!localPts.length) {
      localPts.push([lenM * 0.5, 0], [-lenM * 0.5, -spanM], [-lenM * 0.5, spanM], [0, -spanM], [0, spanM]);
    }
    const cs = Math.cos(pose.ang), sn = Math.sin(pose.ang);
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (let i = 0; i < localPts.length; i++) {
      const pt = localPts[i];
      const wx = pose.x + pt[0] * cs - pt[1] * sn;
      const wy = pose.y + pt[0] * sn + pt[1] * cs;
      if (wx < minX) minX = wx;
      if (wy < minY) minY = wy;
      if (wx > maxX) maxX = wx;
      if (wy > maxY) maxY = wy;
    }
    if (!isFinite(minX) || !isFinite(minY) || !isFinite(maxX) || !isFinite(maxY)) return null;
    return { minX: minX, minY: minY, maxX: maxX, maxY: maxY };
  }
  function syncMarkerFlightBlazerOverlayButton() {
    if (!markerFlightBlazerOverlayBtn || !container) return;
    const sel = state.selectedObject;
    const show = !!(layoutMarkersVisible() && sel && sel.type === 'layoutMarker' && sel.obj && sel.obj.kind === 'flight');
    if (!show) {
      markerFlightBlazerOverlayBtn.style.display = 'none';
      markerFlightHeadingOverlayBtn.style.display = 'none';
      markerFlightBlazerPaletteWrap.style.display = 'none';
      return;
    }
    const mk = sel.obj;
    ensureMarkerFlightBlazerState(mk);
    const b = markerFlightBoundsWorld(mk);
    if (!b) {
      markerFlightBlazerOverlayBtn.style.display = 'none';
      markerFlightHeadingOverlayBtn.style.display = 'none';
      markerFlightBlazerPaletteWrap.style.display = 'none';
      return;
    }
    const sc = worldToScreenCanvas(b.minX, b.minY);
    const left = Math.max(6, sc[0] - 8);
    const top = Math.max(6, sc[1] - 32);
    markerFlightBlazerOverlayBtn.textContent = 'Blazer: ' + (mk.blazerEnabled ? 'ON' : 'OFF');
    markerFlightBlazerOverlayBtn.style.left = left.toFixed(1) + 'px';
    markerFlightBlazerOverlayBtn.style.top = top.toFixed(1) + 'px';
    markerFlightBlazerOverlayBtn.style.display = 'inline-block';
    markerFlightHeadingOverlayBtn.textContent = 'Heading: ' + (mk.headingReversed ? 'REV' : 'FWD');
    markerFlightHeadingOverlayBtn.style.left = (left + 94).toFixed(1) + 'px';
    markerFlightHeadingOverlayBtn.style.top = top.toFixed(1) + 'px';
    markerFlightHeadingOverlayBtn.style.display = 'inline-block';
    markerFlightBlazerPaletteWrap.style.left = left.toFixed(1) + 'px';
    markerFlightBlazerPaletteWrap.style.top = (top + 34).toFixed(1) + 'px';
    markerFlightBlazerPaletteWrap.style.display = 'flex';
    markerFlightBlazerPaletteWrap.querySelectorAll('button[data-blazer-color]').forEach(function(btn) {
      const on = String(btn.getAttribute('data-blazer-color') || '') === String(mk.blazerColor || '');
      btn.style.outline = on ? '2px solid #ffffff' : 'none';
    });
  }
  /** PAPI bar/lamp size vs original (~30% smaller → scale 0.7). */
  const PAPI_VISUAL_SCALE = 0.7;
  /** World units between adjacent PAPI lamp centers (layout coordinates). */
  const PAPI_LAMP_SPACING_WORLD = 14 * PAPI_VISUAL_SCALE;
  function papiLampCenterXsWorld(cx) {
    const sp = PAPI_LAMP_SPACING_WORLD;
    return [cx - 1.5 * sp, cx - 0.5 * sp, cx + 0.5 * sp, cx + 1.5 * sp];
  }
  function layoutMarkerHandleHitRadiusWorld() {
    return Math.max(CELL_SIZE * 0.28, 8 / Math.max(state.scale, 0.1));
  }
  function hitTestLayoutMarkerHandle(wx, wy) {
    if (!layoutMarkersVisible() || state.markerDrawing) return null;
    const sel = state.selectedObject;
    if (!sel || sel.type !== 'layoutMarker' || !sel.obj) return null;
    const mk = sel.obj;
    if (!mk || String(mk.id) !== String(sel.id)) return null;
    const click = [wx, wy];
    const r = layoutMarkerHandleHitRadiusWorld();
    const r2 = r * r;
    if (mk.kind === 'text') {
      const x = Number(mk.x), y = Number(mk.y);
      if (isFinite(x) && isFinite(y) && dist2(click, [x, y]) <= r2)
        return { markerId: mk.id, handle: 'textAnchor' };
    } else if (mk.kind === 'ruler') {
      const x1 = Number(mk.x1), y1 = Number(mk.y1), x2 = Number(mk.x2), y2 = Number(mk.y2);
      if (![x1, y1, x2, y2].every(isFinite)) return null;
      const dA = dist2(click, [x1, y1]);
      const dB = dist2(click, [x2, y2]);
      if (dA <= r2 && dB <= r2) return { markerId: mk.id, handle: dA <= dB ? 'rulerA' : 'rulerB' };
      if (dA <= r2) return { markerId: mk.id, handle: 'rulerA' };
      if (dB <= r2) return { markerId: mk.id, handle: 'rulerB' };
    } else if (mk.kind === 'flight') {
      const pose = resolveMarkerFlightPose(mk);
      if (!pose) return null;
      if (dist2(click, [pose.x, pose.y]) <= r2) return { markerId: mk.id, handle: 'flightCenter' };
    } else if (mk.kind === 'island' || mk.kind === 'area') {
      const pts = mk.points;
      if (!pts || !pts.length) return null;
      let best = null;
      let bestD2 = r2;
      for (let vi = 0; vi < pts.length; vi++) {
        const x = Number(pts[vi].x), y = Number(pts[vi].y);
        if (!isFinite(x) || !isFinite(y)) continue;
        const d2 = dist2(click, [x, y]);
        if (d2 <= bestD2) {
          bestD2 = d2;
          best = { markerId: mk.id, handle: 'islandVertex', vertexIndex: vi };
        }
      }
      return best;
    } else if (mk.kind === 'navaid') {
      const x = Number(mk.x), y = Number(mk.y);
      if (!isFinite(x) || !isFinite(y)) return null;
      const sub = (mk.subType === 'ils') ? 'ils' : 'papi';
      if (sub === 'papi') {
        const xs = papiLampCenterXsWorld(x);
        for (let pi = 0; pi < 4; pi++) {
          if (dist2(click, [xs[pi], y]) <= r2) return { markerId: mk.id, handle: 'navaidCenter' };
        }
        if (dist2(click, [x, y]) <= r2) return { markerId: mk.id, handle: 'navaidCenter' };
        const half = 1.5 * PAPI_LAMP_SPACING_WORLD + r;
        if (Math.abs(click[1] - y) <= r && click[0] >= x - half && click[0] <= x + half)
          return { markerId: mk.id, handle: 'navaidCenter' };
        return null;
      }
      if (dist2(click, [x, y]) <= r2) return { markerId: mk.id, handle: 'navaidCenter' };
    }
    return null;
  }
  function layoutMarkerDrawEndpointDot(ctx, wx, wy, selected) {
    const rad = Math.max(3.5 / Math.max(state.scale, 0.08), CELL_SIZE * 0.11);
    ctx.beginPath();
    ctx.arc(wx, wy, rad, 0, Math.PI * 2);
    ctx.fillStyle = selected ? '#fbbf24' : '#94a3b8';
    ctx.fill();
    ctx.strokeStyle = selected ? '#fffbeb' : 'rgba(15,23,42,0.95)';
    ctx.lineWidth = Math.max(1, 1.35 / Math.max(state.scale, 0.08));
    ctx.stroke();
  }
  function layoutPathDraftStrokeStyle() {
    return 'rgba(148,163,184,0.95)';
  }
  function layoutPathDraftLineWidthPx() {
    return Math.max(1, 1.3 / Math.max(state.scale, 0.1));
  }
  function layoutPathDraftDashPattern() {
    return [5, 5];
  }
  /** @param {number[][]} pts Each [x,y] in layout px. Optional hoverXY draws one more segment from last pt. */
  function strokeLayoutPathDraftPolyline(ctx, pts, hoverXY) {
    if (!pts || pts.length < 1) return;
    const hasHover = hoverXY && hoverXY.length >= 2 && isFinite(hoverXY[0]) && isFinite(hoverXY[1]);
    if (pts.length < 2 && !hasHover) return;
    ctx.save();
    ctx.strokeStyle = layoutPathDraftStrokeStyle();
    ctx.lineWidth = layoutPathDraftLineWidthPx();
    ctx.setLineDash(layoutPathDraftDashPattern());
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
    if (hasHover) ctx.lineTo(hoverXY[0], hoverXY[1]);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.restore();
  }
  function drawLayoutPathDraftVertexDots(ctx, pts, hoverXY) {
    if (!pts) return;
    for (let i = 0; i < pts.length; i++) {
      layoutMarkerDrawEndpointDot(ctx, pts[i][0], pts[i][1], false);
    }
    if (hoverXY && hoverXY.length >= 2 && isFinite(hoverXY[0]) && isFinite(hoverXY[1])) {
      layoutMarkerDrawEndpointDot(ctx, hoverXY[0], hoverXY[1], false);
    }
  }
  function strokeLayoutPathDraftCloseHintArc(ctx, cx, cy, r) {
    ctx.save();
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(56,189,248,0.9)';
    ctx.lineWidth = Math.max(1, 1.2 / Math.max(state.scale, 0.1));
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.restore();
  }
  function layoutMarkerTextHitRect(m) {
    const hx = Number(m.x), hy = Number(m.y);
    if (!isFinite(hx) || !isFinite(hy)) return null;
    const fs = Math.max(10, 12 / Math.max(state.scale, 0.12));
    const txt = String(m.text || '');
    let tw = Math.max(CELL_SIZE * 0.35, txt.length * fs * 0.45) + 8;
    if (ctx) {
      ctx.save();
      ctx.setTransform(1, 0, 0, 1, 0, 0);
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
    if (!state.hasSimulationResult || !state.flights || !state.flights.length) return null;
    const tSec = state.simTimeSec;
    let best = null;
    let bestD2 = (CELL_SIZE * FLIGHT_TOOLTIP_CF) ** 2;
    const flights = state.flights;
    for (let i = 0; i < flights.length; i++) {
      const f = flights[i];
      if (!f || flightBlockedLikeNoWay(f)) continue;
      const pose = getFlightPoseAtTimeForDraw(f, tSec);
      if (!pose) continue;
      const dx = pose.x - wx, dy = pose.y - wy;
      const d2 = dx * dx + dy * dy;
      if (d2 < bestD2) {
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
    return out;
  }
  function pathArcComputePreviewWorldPxFromAB(Ax, Ay, Bx, By, wx, wy) {
    const dx = Bx - Ax, dy = By - Ay;
    const chordLen = Math.hypot(dx, dy) || 1;
    const ex = dx / chordLen, ey = dy / chordLen;
    const nx = -ey, ny = ex;
    const M = [(Ax + Bx) * 0.5, (Ay + By) * 0.5];
    let h = (wx - M[0]) * nx + (wy - M[1]) * ny;
    const maxH = chordLen * PATH_ARC_MAX_BULGE_FRAC;
    h = Math.max(-maxH, Math.min(maxH, h));
    if (Math.abs(h) < PATH_ARC_MIN_BULGE_PX) return [[Ax, Ay], [Bx, By]];
    const Cx = M[0] + nx * h, Cy = M[1] + ny * h;
    return pathArcSampleThreePointWorldPx(Ax, Ay, Bx, By, Cx, Cy, Math.max(CELL_SIZE * 0.28, chordLen / 28));
  }
  function pathArcComputePreviewWorldPx(tw, vertexIndex, wx, wy) {
    if (!tw || tw.pathType === 'runway') return null;
    const verts = tw.vertices;
    if (!verts || vertexIndex <= 0 || vertexIndex >= verts.length - 1) return null;
    const A = cellToPixel(verts[vertexIndex - 1].col, verts[vertexIndex - 1].row);
    const B = cellToPixel(verts[vertexIndex + 1].col, verts[vertexIndex + 1].row);
    return pathArcComputePreviewWorldPxFromAB(A[0], A[1], B[0], B[1], wx, wy);
  }
  function pathArcCommitIslandVertexFromPreview(mk, vertexIndex, previewPx, snapToGrid) {
    if (!mk || !isLayoutPolygonMarkerKind(mk.kind) || !previewPx || previewPx.length < 2) return;
    const verts = mk.points;
    const n = verts ? verts.length : 0;
    if (!verts || n < 3 || vertexIndex < 0 || vertexIndex >= n) return;
    const prev = verts[(vertexIndex - 1 + n) % n];
    const next = verts[(vertexIndex + 1) % n];
    const Apx = [Number(prev.x), Number(prev.y)];
    const Bpx = [Number(next.x), Number(next.y)];
    const workPx = previewPx.map(function(p, j) {
      if (j === 0) return [Apx[0], Apx[1]];
      if (j === previewPx.length - 1) return [Bpx[0], Bpx[1]];
      return [p[0], p[1]];
    });
    const cells = [];
    if (previewPx.length > 2) {
      const densePx = pathArcDensifyPolylinePx(workPx, Math.max(3, CELL_SIZE * 0.11));
      for (let k = 0; k < densePx.length; k++) {
        const snap = worldPointToPixel(densePx[k][0], densePx[k][1], snapToGrid);
        const c = { x: snap[0], y: snap[1] };
        if (cells.length && cells[cells.length - 1].x === c.x && cells[cells.length - 1].y === c.y) continue;
        if (c.x === prev.x && c.y === prev.y) continue;
        cells.push(c);
      }
      while (cells.length && cells[cells.length - 1].x === next.x && cells[cells.length - 1].y === next.y) cells.pop();
    }
    if (!cells.length) {
      const M = [(Apx[0] + Bpx[0]) * 0.5, (Apx[1] + Bpx[1]) * 0.5];
      const snap = worldPointToPixel(M[0], M[1], snapToGrid);
      cells.push({ x: snap[0], y: snap[1] });
    }
    const newArr = verts.slice();
    newArr.splice(vertexIndex, 1, ...cells);
    mk.points = newArr;
    const midSel = vertexIndex + Math.max(0, Math.floor((cells.length - 1) / 2));
    state.selectedVertex = { type: 'layoutMarkerHandle', id: mk.id, handle: 'islandVertex', vertexIndex: midSel };
  }
  function pathArcCommitFromPreview(tw, vertexIndex, previewPx, snapToGrid) {
    if (!tw || tw.pathType === 'runway') return;
    if (!previewPx || previewPx.length < 2) return;
    const verts = tw.vertices;
    if (!verts || vertexIndex <= 0 || vertexIndex >= verts.length - 1) return;
    const prev = verts[vertexIndex - 1], next = verts[vertexIndex + 1];
    const Apx = cellToPixel(prev.col, prev.row);
    const Bpx = cellToPixel(next.col, next.row);
