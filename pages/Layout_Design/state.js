  function c2dTaxiwayPavementFill() {
    const s = _canvas2dStyle.taxiwayPavementFill;
    return (typeof s === 'string' && s.trim()) ? s.trim() : '#908e82';
  }
  function c2dRunwayOutline() { return _canvas2dStyle.runwayOutline || '#cbd5e1'; }
  function c2dRunwayMarkingColor() { return _canvas2dStyle.runwayMarkingColor || '#f8fafc'; }
  function c2dRunwayThresholdColor() { return _canvas2dStyle.runwayThresholdColor || c2dRunwayMarkingColor(); }
  function c2dRunwayCenterlineColor() { return _canvas2dStyle.runwayCenterlineColor || c2dRunwayMarkingColor(); }
  function c2dRunwayTouchdownColor() { return _canvas2dStyle.runwayTouchdownColor || c2dRunwayMarkingColor(); }
  function c2dRunwayAimingPointColor() { return _canvas2dStyle.runwayAimingPointColor || c2dRunwayMarkingColor(); }
  function c2dRunwayExtensionFill() { return _canvas2dStyle.runwayExtensionFill || c2dRunwayStroke(); }
  function c2dRunwayBlastChevronColor() { return _canvas2dStyle.runwayBlastChevronColor || '#facc15'; }
  /** Strip alpha from rgba for solid road surface when showRoadWidth is on. */
  function c2dCssColorToOpaque(css) {
    const s = String(css || '').trim();
    const ra = s.match(/^rgba\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*[\d.]+\s*\)/i);
    if (ra) return 'rgb(' + ra[1] + ',' + ra[2] + ',' + ra[3] + ')';
    return s;
  }
  const C2D_COLOR_SHADE_STEP_MUL = 0.88;
  function c2dParseCssRgbTriplet(css) {
    const s = String(css || '').trim();
    let m = s.match(/^#([0-9a-f]{3})$/i);
    if (m) {
      const h = m[1];
      return [parseInt(h[0] + h[0], 16), parseInt(h[1] + h[1], 16), parseInt(h[2] + h[2], 16)];
    }
    m = s.match(/^#([0-9a-f]{6})$/i);
    if (m) {
      const h = m[1];
      return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
    }
    m = s.match(/^rgba?\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)/i);
    if (m) return [Number(m[1]), Number(m[2]), Number(m[3])];
    return null;
  }
  function c2dCssColorLightenSteps(css, steps) {
    const opaque = c2dCssColorToOpaque(css);
    const t = c2dParseCssRgbTriplet(opaque);
    const n = Number(steps);
    if (!t || !(n > 0)) return opaque;
    const f = Math.pow(1 / C2D_COLOR_SHADE_STEP_MUL, n);
    const r = Math.max(0, Math.min(255, Math.round(t[0] * f)));
    const g = Math.max(0, Math.min(255, Math.round(t[1] * f)));
    const b = Math.max(0, Math.min(255, Math.round(t[2] * f)));
    return 'rgb(' + r + ',' + g + ',' + b + ')';
  }
  /** Multiply RGB channels (e.g. 0.99 ≈ 1% darker). Expects opaque-ish CSS; alpha stripped first. */
  function c2dCssColorRgbChannelScale(css, mul) {
    const opaque = c2dCssColorToOpaque(css);
    const t = c2dParseCssRgbTriplet(opaque);
    const f = Number(mul);
    if (!t || !isFinite(f)) return opaque;
    const r = Math.max(0, Math.min(255, Math.round(t[0] * f)));
    const g = Math.max(0, Math.min(255, Math.round(t[1] * f)));
    const b = Math.max(0, Math.min(255, Math.round(t[2] * f)));
    return 'rgb(' + r + ',' + g + ',' + b + ')';
  }
  /** Same rgb as layout marker kind=area fill (`drawLayoutAreaMarkers2DFloor`, 3 lighten steps). */
  function c2dRoadWidthBandSurfaceColor() {
    return c2dCssColorLightenSteps(c2dRunwayStroke(), 3);
  }
  /** Taxiway / apron taxiway width band: one step darker than marker area (2 lighten steps vs runway stroke). */
  function c2dRoadWidthBandTaxiwaySurfaceColor() {
    return c2dCssColorLightenSteps(c2dRunwayStroke(), 2);
  }
  /** Runway path & runway taxiway (runway_exit) width band: dark asphalt gray. */
  function c2dRoadWidthBandRunwayAsphaltColor() {
    return '#363636';
  }
  /** Layer mono: cool blue-gray (slate), not warm taupe. */
  function c2dLayerMonoLineStrokeCss() {
    return '#94a3b8';
  }
  /** Layer mono: fills match path **asphalt** width band (`c2dRoadWidthBandForPavement('asphalt')`), not cement / `c2dTaxiwayPavementFill`. */
  function c2dLayerMonoFillDarkAsphaltCss() {
    return c2dCssColorToOpaque(c2dRoadWidthBandRunwayAsphaltColor());
  }
  function c2dLayerMonoFillDarkAsphaltRgba(a) {
    const t = c2dParseCssRgbTriplet(c2dLayerMonoFillDarkAsphaltCss());
    const al = Number(a);
    if (!t || !isFinite(al)) return c2dLayerMonoFillDarkAsphaltCss();
    return 'rgba(' + t[0] + ',' + t[1] + ',' + t[2] + ',' + Math.max(0, Math.min(1, al)) + ')';
  }
  const C2D_LAYER_MONO_ETC_WHITE = '#f8fafc';
  function pathPavementDefaultForPathType(pathType) {
    const pt = pathType || 'taxiway';
    if (pt === 'runway' || pt === 'runway_exit') return 'asphalt';
    return 'cement';
  }
  function pathPavementResolvedForTaxiway(tw) {
    if (!tw || typeof tw !== 'object') return 'cement';
    const v = tw.pavement;
    if (v === 'asphalt' || v === 'cement') return v;
    return pathPavementDefaultForPathType(tw.pathType);
  }
  function c2dRoadWidthBandForPavement(pavement) {
    return pavement === 'cement' ? c2dRoadWidthBandTaxiwaySurfaceColor() : c2dRoadWidthBandRunwayAsphaltColor();
  }
  function normalizePathPavementInPlace(tw) {
    if (!tw || typeof tw !== 'object') return;
    const v = tw.pavement;
    if (v === 'asphalt' || v === 'cement') return;
    tw.pavement = pathPavementDefaultForPathType(tw.pathType);
  }
  function islandMarkerPavementResolved(m) {
    if (!m || typeof m !== 'object') return 'asphalt';
    const pv = m.pavement;
    if (pv === 'asphalt' || pv === 'cement') return pv;
    const op = m.outerPavement;
    if (op === 'taxiway') return 'cement';
    if (op === 'runway') return 'asphalt';
    return 'asphalt';
  }
  function islandMarkerPavementFillCss(m) {
    return c2dRoadWidthBandForPavement(islandMarkerPavementResolved(m));
  }
  /** Selected road-width band only: ~1% darker than theme stroke/fill. */
  const ROAD_WIDTH_SURFACE_RGB_MUL = 0.99;
  function c2dObjectSelectedGlowBlur() {
    const n = Number(_canvas2dStyle.objectSelectedGlowBlur);
    return (isFinite(n) && n >= 0) ? n : 22;
  }
  function c2dFlightSelectedRingStroke() { return _canvas2dStyle.flightSelectedRingStroke || '#facc15'; }
  function c2dFlightSelectedRingGlow() { return _canvas2dStyle.flightSelectedRingGlow || 'rgba(250, 204, 21, 0.55)'; }
  function c2dFlightSelectedRingGlowBlur() {
    const n = Number(_canvas2dStyle.flightSelectedRingGlowBlur);
    return (isFinite(n) && n >= 0) ? n : 18;
  }
  function c2dSimPreTouchdownHaloStroke() { return _canvas2dStyle.simPreTouchdownHaloStroke || 'rgba(239, 68, 68, 0.92)'; }
  function c2dSimPreTouchdownHaloFill() { return _canvas2dStyle.simPreTouchdownHaloFill || 'rgba(239, 68, 68, 0.18)'; }
  function c2dSimPreTouchdownHaloBlur() {
    const n = Number(_canvas2dStyle.simPreTouchdownHaloBlur);
    return (isFinite(n) && n >= 0) ? n : 14;
  }
  function c2dSimFlightTrailStroke() { return _canvas2dStyle.simFlightTrailStroke || 'rgba(255, 47, 146, 0.97)'; }
  function c2dSimFlightTrailStrokeEnd() { return _canvas2dStyle.simFlightTrailStrokeEnd || 'rgba(255, 47, 146, 0)'; }
  function c2dSimFlightTrailLineWidth() {
    const n = Number(_canvas2dStyle.simFlightTrailLineWidth);
    return (isFinite(n) && n > 0) ? n : 3.5;
  }
  function c2dApproachPreviewWidthM() {
    const n = Number(_canvas2dStyle.approachPreviewWidthM);
    return (isFinite(n) && n > 0) ? n : 30;
  }
  function c2dApproachPreviewStroke() {
    return _canvas2dStyle.approachPreviewStroke || 'rgba(255, 255, 255, 0.01)';
  }
  function c2dHoldingPointDiameterM() {
    const n = Number(_canvas2dStyle.holdingPointDiameterM);
    return (isFinite(n) && n > 0) ? n : 15;
  }
  function normalizeHoldingPointKind(raw) {
    return raw === 'runway_holding' ? 'runway_holding' : 'intermediate';
  }
  function pathTypeToHpKind(pathType) {
    return pathType === 'runway_exit' ? 'runway_holding' : 'intermediate';
  }
  function holdingPointKindDisplayLabel(kind) {
    return normalizeHoldingPointKind(kind) === 'runway_holding' ? 'Runway Holding Position' : 'Intermediate Holding Position';
  }
  function c2dHoldingPointFillForKind(kind) {
    const k = normalizeHoldingPointKind(kind);
    if (k === 'runway_holding') return _canvas2dStyle.holdingPointRunwayFill || 'rgba(239, 68, 68, 0.5)';
    return _canvas2dStyle.holdingPointIntermediateFill || 'rgba(249, 115, 22, 0.5)';
  }
  function c2dHoldingPointStrokeForKind(kind) {
    const k = normalizeHoldingPointKind(kind);
    if (k === 'runway_holding') return _canvas2dStyle.holdingPointRunwayStroke || 'rgba(220, 38, 38, 0.78)';
    return _canvas2dStyle.holdingPointIntermediateStroke || 'rgba(234, 88, 12, 0.75)';
  }
  function c2dHoldingPointPreviewFillForPathType(pathType) {
    const k = pathTypeToHpKind(pathType || 'taxiway');
    if (k === 'runway_holding') return _canvas2dStyle.holdingPointRunwayPreviewFill || 'rgba(239, 68, 68, 0.28)';
    return _canvas2dStyle.holdingPointIntermediatePreviewFill || 'rgba(249, 115, 22, 0.28)';
  }
  function c2dHoldingPointPreviewStrokeForPathType(pathType) {
    const k = pathTypeToHpKind(pathType || 'taxiway');
    if (k === 'runway_holding') return _canvas2dStyle.holdingPointRunwayStroke || 'rgba(220, 38, 38, 0.78)';
    return _canvas2dStyle.holdingPointIntermediateStroke || 'rgba(234, 88, 12, 0.75)';
  }
  function c2dHoldingPointMarkingYellow() {
    return _canvas2dStyle.holdingPointMarkingYellow || '#facc15';
  }
  function c2dHoldingPointMarkingLineWidthWorld() {
    const n = Number(_canvas2dStyle.holdingPointMarkingLineWidthWorld);
    return (isFinite(n) && n > 0) ? n : 0.28;
  }
  function holdingPointMarkingDoubleLineGapM(lineW) {
    const n = Number(_canvas2dStyle.holdingPointMarkingDoubleLineGapM);
    const lw = Number(lineW);
    const baseLw = (isFinite(lw) && lw > 0) ? lw : c2dHoldingPointMarkingLineWidthWorld();
    return (isFinite(n) && n > 0) ? n : Math.max(0.28, baseLw * 1.2);
  }
  function taxiwayWorldWidthMForHolding(tw) {
    if (!tw) return TAXIWAY_DEFAULT_WIDTH;
    const typ = tw.pathType || 'taxiway';
    const base = typ === 'runway' ? RUNWAY_PATH_DEFAULT_WIDTH : (typ === 'runway_exit' ? RUNWAY_EXIT_DEFAULT_WIDTH : TAXIWAY_DEFAULT_WIDTH);
    return clampTaxiwayWidthM(typ, tw.width, base);
  }
  function holdingPointBarHalfLengthMFromPathWidth(pathWidthM) {
    const w = Number(pathWidthM);
    if (isFinite(w) && w > 0) return w * 0.5;
    return Math.max(3, c2dHoldingPointDiameterM() * 0.5);
  }
  function holdingPointPerpFromTangent(ux, uy) {
    return { px: -uy, py: ux };
  }
  function distPointToSegmentSq(x, y, ax, ay, bx, by) {
    const abx = bx - ax, aby = by - ay;
    const apx = x - ax, apy = y - ay;
    const abLenSq = abx * abx + aby * aby;
    if (abLenSq < 1e-12) return apx * apx + apy * apy;
    let t = (apx * abx + apy * aby) / abLenSq;
    t = Math.max(0, Math.min(1, t));
    const qx = ax + t * abx, qy = ay + t * aby;
    const dx = x - qx, dy = y - qy;
    return dx * dx + dy * dy;
  }
  function findHoldingPointPathGeometry(hp) {
    const pt = [hp.x, hp.y];
    const wantRunway = normalizeHoldingPointKind(hp.hpKind) === 'runway_holding';
    const maxD2 = Math.pow(Math.max(CELL_SIZE * 6, 55), 2);
    let bestD2 = Infinity;
    let ux = 1, uy = 0;
    let bestTw = null;
    (state.taxiways || []).forEach(function(tw) {
      const typ = tw.pathType || 'taxiway';
      if (wantRunway) {
        if (typ !== 'runway_exit') return;
      } else {
        if (typ !== 'taxiway' && typ !== 'apron_taxiway' && typ !== 'general_queue_taxiway') return;
      }
      if (!tw.vertices || tw.vertices.length < 2) return;
      for (let i = 0; i < tw.vertices.length - 1; i++) {
        const p1 = cellToPixel(tw.vertices[i].col, tw.vertices[i].row);
        const p2 = cellToPixel(tw.vertices[i + 1].col, tw.vertices[i + 1].row);
        const near = closestPointOnSegment(p1, p2, pt);
        if (!near) continue;
        const d2 = dist2(near, pt);
        if (d2 < bestD2) {
          bestD2 = d2;
          bestTw = tw;
          const dx = p2[0] - p1[0], dy = p2[1] - p1[1];
          const len = Math.hypot(dx, dy);
          if (len > 1e-6) {
            ux = dx / len;
            uy = dy / len;
          }
        }
      }
    });
    const pathWidthM = taxiwayWorldWidthMForHolding(bestTw);
    if (bestD2 > maxD2) return { ux: 1, uy: 0, ok: false, pathWidthM, tw: bestTw };
    return { ux, uy, ok: true, pathWidthM, tw: bestTw };
  }
  function closestPointOnAnyRunwayCenterlineWorld(wx, wy) {
    const pt = [wx, wy];
    let best = null;
    let bestD2 = Infinity;
    (state.taxiways || []).forEach(function(tw) {
      if ((tw.pathType || 'taxiway') !== 'runway') return;
      if (!tw.vertices || tw.vertices.length < 2) return;
      for (let i = 0; i < tw.vertices.length - 1; i++) {
        const p1 = cellToPixel(tw.vertices[i].col, tw.vertices[i].row);
        const p2 = cellToPixel(tw.vertices[i + 1].col, tw.vertices[i + 1].row);
        const near = closestPointOnSegment(p1, p2, pt);
        if (!near) continue;
        const d2 = dist2(near, pt);
        if (d2 < bestD2) { bestD2 = d2; best = near; }
      }
    });
    return best;
  }
  function findHoldingPointPathTangent(hp) {
    const g = findHoldingPointPathGeometry(hp);
    return { ux: g.ux, uy: g.uy, ok: g.ok };
  }
  function drawHoldingPointGridMarking(ctx, cx, cy, hpKind, selected, preview) {
    const k = normalizeHoldingPointKind(hpKind);
    const g = findHoldingPointPathGeometry({ x: cx, y: cy, hpKind: hpKind });
    const { px, py } = holdingPointPerpFromTangent(g.ux, g.uy);
    const halfLen = holdingPointBarHalfLengthMFromPathWidth(g.pathWidthM);
    const pathSpanM = halfLen * 2;
    const lineW = c2dHoldingPointMarkingLineWidthWorld();
    const centerlineStroke = k === 'runway_holding' ? c2dRunwayTaxiwayCenterlineStroke() : c2dTaxiwayCenterlineStroke();
    const lineMono = layerMonoLinesOn() && !preview && !selected;
    const stroke = preview
      ? 'rgba(250, 204, 21, 0.7)'
      : (selected ? c2dObjectSelectedStroke() : (lineMono ? c2dLayerMonoLineStrokeCss() : centerlineStroke));
