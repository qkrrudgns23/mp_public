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
    const stroke = preview ? 'rgba(250, 204, 21, 0.7)' : (selected ? c2dObjectSelectedStroke() : centerlineStroke);
    const lw = preview ? Math.max(0.2, lineW * 0.92) : (selected ? lineW + 0.14 : lineW);
    const pairHalf = holdingPointMarkingDoubleLineGapM(lineW) * 0.5;
    const dashLen = Math.max(lineW * 2.2, pathSpanM * 0.13);
    const gapLen = Math.max(lineW * 1.6, pathSpanM * 0.09);
    ctx.lineCap = 'butt';
    ctx.lineJoin = 'miter';
    ctx.strokeStyle = stroke;
    ctx.lineWidth = lw;
    if (selected && !preview) {
      ctx.shadowColor = c2dObjectSelectedGlow();
      ctx.shadowBlur = c2dObjectSelectedGlowBlur();
    } else {
      ctx.shadowBlur = 0;
    }
    function strokeBarAtOffset(ofs) {
      const sx = cx - px * halfLen + g.ux * ofs;
      const sy = cy - py * halfLen + g.uy * ofs;
      const ex = cx + px * halfLen + g.ux * ofs;
      const ey = cy + py * halfLen + g.uy * ofs;
      ctx.beginPath();
      ctx.moveTo(sx, sy);
      ctx.lineTo(ex, ey);
      ctx.stroke();
    }
    if (k === 'intermediate') {
      ctx.setLineDash([dashLen, gapLen]);
      strokeBarAtOffset(0);
      ctx.setLineDash([]);
    } else {
      ctx.setLineDash([]);
      strokeBarAtOffset(-pairHalf);
      strokeBarAtOffset(pairHalf);
      const R = closestPointOnAnyRunwayCenterlineWorld(cx, cy);
      const rx = R ? R[0] : cx + g.ux * (CELL_SIZE * 40);
      const ry = R ? R[1] : cy + g.uy * (CELL_SIZE * 40);
      const midM = [cx - g.ux * pairHalf, cy - g.uy * pairHalf];
      const midP = [cx + g.ux * pairHalf, cy + g.uy * pairHalf];
      const ofsR = dist2(midM, [rx, ry]) <= dist2(midP, [rx, ry]) ? -pairHalf : pairHalf;
      const pathW = Number(g.pathWidthM);
      const toothLen = Math.max(0.75, (isFinite(pathW) && pathW > 0 ? pathW : 12) * 0.24) * 0.25;
      const toothSpacing = Math.max(0.55, pathSpanM * 0.065);
      const toothLw = Math.max(lw, lw * 1.12);
      ctx.save();
      ctx.lineWidth = toothLw;
      for (let s = -halfLen + toothSpacing * 0.5; s <= halfLen - toothSpacing * 0.25; s += toothSpacing) {
        const bx = cx + px * s + g.ux * ofsR;
        const by = cy + py * s + g.uy * ofsR;
        const mx = cx + px * s;
        const my = cy + py * s;
        const vx = rx - mx;
        const vy = ry - my;
        const signT = (g.ux * vx + g.uy * vy) >= 0 ? 1 : -1;
        ctx.beginPath();
        ctx.moveTo(bx, by);
        ctx.lineTo(bx + g.ux * signT * toothLen, by + g.uy * signT * toothLen);
        ctx.stroke();
      }
      ctx.restore();
    }
    ctx.shadowBlur = 0;
  }
  function c2dSimStandOccupiedFill() { return _canvas2dStyle.simStandOccupiedFill || 'rgba(239, 68, 68, 0.32)'; }
  function c2dSimStandOccupiedStroke() { return _canvas2dStyle.simStandOccupiedStroke || 'rgba(220, 38, 38, 0.95)'; }
  function c2dPathDrawStartMarkerRadiusPx() {
    const n = Number(_canvas2dStyle.pathDrawStartMarkerRadiusPx);
    const base = (isFinite(n) && n > 0) ? n : 3.5;
    return base * LAYOUT_VERTEX_DOT_SCALE;
  }
  function c2dPathDrawStartMarkerStrokePx() {
    const n = Number(_canvas2dStyle.pathDrawStartMarkerStrokePx);
    const base = (isFinite(n) && n > 0) ? n : 1;
    return Math.max(0.5, base * LAYOUT_VERTEX_DOT_SCALE);
  }
  function c2dPathDrawStartLabelFontPx() {
    const n = Number(_canvas2dStyle.pathDrawStartLabelFontPx);
    const base = (isFinite(n) && n >= 6) ? n : 8;
    return Math.max(6, Math.round(base * LAYOUT_VERTEX_DOT_SCALE));
  }
  function c2dPathDrawStartLabelOffsetY() {
    const n = Number(_canvas2dStyle.pathDrawStartLabelOffsetY);
    const base = isFinite(n) ? n : -6;
    return base * LAYOUT_VERTEX_DOT_SCALE;
  }
  const GANTT_COLORS = {
    S_BAR: _ganttStyle.sBar || '#007aff',
    S_SERIES: _ganttStyle.sSeries || '#38bdf8',
    E_BAR: _ganttStyle.eBar || '#fb37c5',
    E_SERIES: _ganttStyle.eSeries || '#fb923c',
