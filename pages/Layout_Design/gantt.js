  function layerIslandAreaFillEffective() {
    if (state.isPanning) return false;
    if (state.markerDrawing && getMarkerSubKindFromPanel() === 'island') return true;
    if (state.markerDrawing && getMarkerSubKindFromPanel() === 'area') return true;
    return !!state.layers.islandAreaFill;
  }
  function layerIslandContourLinesEffective() {
    if (settingModeSelect && settingModeSelect.value === 'marker') return true;
    return !!state.layers.islandAreaLines;
  }
  function layoutMarkersVisible() {
    if (settingModeSelect && settingModeSelect.value === 'marker') return true;
    if (state.markerDrawing) return true;
    return !!(state.layers.textRuler || state.layers.dummyFlight || state.layers.islandAreaLines || state.layers.islandAreaFill);
  }
  function getMarkerSubKindFromPanel() {
    const tab = document.querySelector('.marker-tool-tab[aria-selected="true"]');
    const v = tab && tab.getAttribute('data-marker-sub');
    return v === 'ruler' || v === 'flight' || v === 'island' || v === 'area' || v === 'navaid' ? v : 'text';
  }
  function getMarkerNavaidTypeFromPanel() {
    const sel = document.getElementById('markerNavaidType');
    const v = String(sel && sel.value || '').trim().toLowerCase();
    return v === 'ils' ? 'ils' : 'papi';
  }
  function syncMarkerNavaidRowVisibility() {
    const row = document.getElementById('markerNavaidRow');
    if (!row) return;
    const show = getMarkerSubKindFromPanel() === 'navaid';
    row.hidden = !show;
    row.style.display = show ? '' : 'none';
  }
  function getMarkerFlightAircraftTypeFromPanel() {
    const sel = document.getElementById('markerFlightAircraftType');
    const v = String(sel && sel.value || '').trim();
    if (v && AIRCRAFT_BY_ID[v]) return v;
    return (AIRCRAFT_TYPES[0] && (AIRCRAFT_TYPES[0].id || AIRCRAFT_TYPES[0].name)) || 'A320';
  }
  function populateMarkerFlightAircraftSelect() {
    const sel = document.getElementById('markerFlightAircraftType');
    if (!sel) return;
    const html = AIRCRAFT_TYPES.map(function(a) {
      const id = String(a.id || a.name || '').trim();
      if (!id) return '';
      const name = String(a.name || a.id || id).trim();
      const icao = String(a.icao || 'C').toUpperCase();
      return '<option value="' + escapeAttr(id) + '">' + escapeHtml(name + ' (ICAO ' + icao + ')') + '</option>';
    }).filter(Boolean).join('');
    sel.innerHTML = html || '<option value="A320">Airbus A320 (ICAO C)</option>';
    if (sel.options.length) sel.value = sel.options[0].value;
  }
  function syncMarkerFlightAircraftRowVisibility() {
    const row = document.getElementById('markerFlightAircraftRow');
    if (!row) return;
    const show = getMarkerSubKindFromPanel() === 'flight';
    row.hidden = !show;
    row.style.display = show ? '' : 'none';
  }
  function syncMarkerIslandWidthRowVisibility() {
    const row = document.getElementById('markerIslandWidthRow');
    if (!row) return;
    const show = getMarkerSubKindFromPanel() === 'island';
    row.hidden = !show;
    row.style.display = show ? '' : 'none';
  }
  function setMarkerSubKindTab(sub) {
    const allowed = { ruler: 1, flight: 1, island: 1, area: 1, navaid: 1 };
    const next = allowed[sub] ? sub : 'text';
    document.querySelectorAll('.marker-tool-tab').forEach(function(btn) {
      const on = (btn.getAttribute('data-marker-sub') || '') === next;
      btn.classList.toggle('active', on);
      btn.setAttribute('aria-selected', on ? 'true' : 'false');
    });
    syncMarkerFlightAircraftRowVisibility();
    syncMarkerIslandWidthRowVisibility();
    syncMarkerNavaidRowVisibility();
  }
  function syncMarkerSubKindTabFromSelectedLayoutMarker() {
    const sel = state.selectedObject;
    if (!sel || sel.type !== 'layoutMarker' || !sel.obj) return;
    const kind = String(sel.obj.kind || '').trim();
    if (kind !== 'text' && kind !== 'ruler' && kind !== 'island' && kind !== 'area' && kind !== 'flight' && kind !== 'navaid') return;
    setMarkerSubKindTab(kind);
    if (kind === 'navaid') {
      const sel2 = document.getElementById('markerNavaidType');
      if (sel2) {
        const sub = (sel.obj.subType === 'ils') ? 'ils' : 'papi';
        sel2.value = sub;
      }
    }
  }
  function isMarkerFlightAllowedPathType(pt) {
    return pt === 'runway' || pt === 'runway_exit' || pt === 'taxiway' || pt === 'general_queue_taxiway';
  }
  function snapWorldToMarkerFlightTaxiway(wx, wy, opts) {
    const click = [wx, wy];
    const o = opts || {};
    const lockTaxiwayId = o.taxiwayId != null ? String(o.taxiwayId) : null;
    const allowFar = o.allowFar === true;
    let best = null;
    let bestD2 = Infinity;
    const maxD2 = allowFar ? Infinity : Math.pow(CELL_SIZE * HIT_TW_SEG_CF, 2);
    (state.taxiways || []).forEach(function(tw) {
      if (!tw || !isMarkerFlightAllowedPathType(tw.pathType || 'taxiway')) return;
      if (lockTaxiwayId && String(tw.id) !== lockTaxiwayId) return;
      const pts = typeof getOrderedPoints === 'function' ? getOrderedPoints(tw) : getTaxiwayOrderedPoints(tw);
      if (!pts || pts.length < 2) return;
      for (let i = 0; i < pts.length - 1; i++) {
        const near = closestPointOnSegment(pts[i], pts[i + 1], click);
        if (!near) continue;
        const d2 = dist2(near, click);
        if (d2 < bestD2 && d2 <= maxD2) {
          bestD2 = d2;
          const ax = pts[i][0], ay = pts[i][1], bx = pts[i + 1][0], by = pts[i + 1][1];
          const dx = bx - ax, dy = by - ay;
          const segLen2 = dx * dx + dy * dy;
          const t = segLen2 < 1e-12 ? 0.5 : Math.max(0, Math.min(1, ((near[0] - ax) * dx + (near[1] - ay) * dy) / segLen2));
          best = { taxiwayId: tw.id, segIndex: i, t: t };
        }
      }
    });
    return best;
  }
  function resolveMarkerFlightPose(m) {
    if (!m || m.kind !== 'flight') return null;
    const tw = (state.taxiways || []).find(function(x) { return x && x.id === m.taxiwayId; });
    if (!tw) return null;
    const pts = typeof getOrderedPoints === 'function' ? getOrderedPoints(tw) : getTaxiwayOrderedPoints(tw);
    if (!pts || pts.length < 2) return null;
    let si = typeof m.segIndex === 'number' && isFinite(m.segIndex) ? Math.floor(m.segIndex) : (parseInt(m.segIndex, 10) || 0);
    si = Math.max(0, Math.min(si, pts.length - 2));
    let t = Number(m.t);
    if (!isFinite(t)) t = 0.5;
    t = Math.max(0, Math.min(1, t));
    const a = pts[si], b = pts[si + 1];
    const x = a[0] + t * (b[0] - a[0]);
    const y = a[1] + t * (b[1] - a[1]);
    let ang = Math.atan2(b[1] - a[1], b[0] - a[0]);
    if (m.headingReversed === true) ang += Math.PI;
    return { x: x, y: y, ang: ang };
  }
  function getMarkerFlightWingtipWorldPoints(m) {
    if (!m || m.kind !== 'flight') return null;
    const pose = resolveMarkerFlightPose(m);
    if (!pose) return null;
    const ac = getAircraftInfoByType(m.aircraftType);
    const lenM = ac && isFinite(Number(ac.length_m)) ? Math.max(1, Number(ac.length_m)) : 40;
    const spanM = ac && isFinite(Number(ac.wingspan_m)) ? Math.max(1, Number(ac.wingspan_m)) : 40;
    const silhouette2D = getApronAircraftDetailedSilhouettePoints();
    let leftLocal = [0, -spanM];
    let rightLocal = [0, spanM];
    if (_ac2d.useDetailedSilhouette === true && silhouette2D.length >= 3) {
      let minY = Infinity, maxY = -Infinity;
      let minPt = null, maxPt = null;
      for (let i = 0; i < silhouette2D.length; i++) {
        const p = silhouette2D[i];
        if (!p || p.length < 2) continue;
        const lx = Number(p[0]) * lenM;
        const ly = Number(p[1]) * spanM;
        if (!isFinite(lx) || !isFinite(ly)) continue;
        if (ly < minY) { minY = ly; minPt = [lx, ly]; }
        if (ly > maxY) { maxY = ly; maxPt = [lx, ly]; }
      }
      if (minPt && maxPt) {
        leftLocal = minPt;
        rightLocal = maxPt;
      }
    }
    const cs = Math.cos(pose.ang), sn = Math.sin(pose.ang);
    function localToWorld(pt) {
      return {
        x: pose.x + pt[0] * cs - pt[1] * sn,
        y: pose.y + pt[0] * sn + pt[1] * cs,
      };
    }
    return { left: localToWorld(leftLocal), right: localToWorld(rightLocal) };
  }
  function ensureMarkerFlightBlazerState(m) {
    if (!m || m.kind !== 'flight') return;
    if (typeof m.blazerEnabled !== 'boolean') m.blazerEnabled = false;
    if (typeof m.headingReversed !== 'boolean') m.headingReversed = false;
    if (MARKER_BLAZER_COLOR_OPTIONS.indexOf(String(m.blazerColor || '').trim()) < 0) m.blazerColor = MARKER_BLAZER_COLOR_OPTIONS[0];
    if (!Array.isArray(m.blazerLeftTrail)) m.blazerLeftTrail = [];
    if (!Array.isArray(m.blazerRightTrail)) m.blazerRightTrail = [];
  }
  function appendMarkerFlightBlazerTrail(m) {
    if (!m || m.kind !== 'flight') return;
    ensureMarkerFlightBlazerState(m);
    if (!m.blazerEnabled) return;
    const tips = getMarkerFlightWingtipWorldPoints(m);
    if (!tips || !tips.left || !tips.right) return;
    const minStep = Math.max(0.25, CELL_SIZE * 0.03);
    const minStep2 = minStep * minStep;
    function append(trail, pt) {
      const last = trail.length ? trail[trail.length - 1] : null;
      if (!last || dist2([last.x, last.y], [pt.x, pt.y]) >= minStep2) trail.push({ x: pt.x, y: pt.y });
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
