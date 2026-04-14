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
    const markYellow = c2dHoldingPointMarkingYellow();
    const stroke = preview ? 'rgba(250, 204, 21, 0.7)' : (selected ? c2dObjectSelectedStroke() : markYellow);
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
    CONFLICT: _ganttStyle.conflict || '#7f1d1d',
    SELECTED: _ganttStyle.selected || '#fbbf24',
  };
  const _apronAc = _layoutTier.apronAircraft || {};
  const _acScaleByCat = (_apronAc.scaleByIcaoCategory && typeof _apronAc.scaleByIcaoCategory === 'object') ? _apronAc.scaleByIcaoCategory : {};
  function apronAircraftScaleForIcao(code) {
    const c = String(code || '').toUpperCase();
    const v = Number(_acScaleByCat[c]);
    if (isFinite(v) && v > 0) return v;
    const d = Number(_acScaleByCat.default);
    return (isFinite(d) && d > 0) ? d : 1.0;
  }
  const _ac2d = _apronAc.twoD || {};
  const _acSil = (_ac2d.silhouette && typeof _ac2d.silhouette === 'object') ? _ac2d.silhouette : {};
  function apron2DGlyphFill() { return _ac2d.fillColor || '#ff2f92'; }
  function getApronAircraftDetailedSilhouettePoints() {
    const raw = _ac2d.detailedSilhouettePoints;
    if (!Array.isArray(raw) || raw.length < 3) return [];
    const out = [];
    for (let i = 0; i < raw.length; i++) {
      const row = raw[i];
      if (!Array.isArray(row) || row.length < 2) continue;
      const x = Number(row[0]);
      const y = Number(row[1]);
      if (isFinite(x) && isFinite(y)) out.push([x, y]);
    }
    return out.length >= 3 ? out : [];
  }
  const _schedAlgo = _algoTier.scheduledTimes || {};
  const SCHED_DWELL_FLOOR_MIN = (function() {
    const v = Number(_schedAlgo.dwellFloorMin);
    return (isFinite(v) && v >= 0) ? v : 20;
  })();
  /** Dispatched schedule (d): SLDT(d) = SIBT(d) − this many minutes; STOT(d) = SOBT(d) + SCHED_SD_STOT_PLUS_SOBD_MIN. */
  const SCHED_SD_SIBT_MINUS_SLD_MIN = 3;
  const SCHED_SD_STOT_PLUS_SOBD_MIN = 3;
  const RSEP_MISSING_MATRIX_SEC = (function() {
    const v = Number(_schedAlgo.rsepMissingMatrixSeparationSec);
    return (isFinite(v) && v >= 0) ? v : 90;
  })();
  const TIME_AXIS_CFG = _algoTier.timeAxis || {};
  const DOM_OPT_CFG = (_algoTier.domOptimization && typeof _algoTier.domOptimization === 'object') ? _algoTier.domOptimization : {};
  const DOM_OPT_FLIGHT_VIRT_ENABLE = DOM_OPT_CFG.flightListVirtualScroll !== false;
  const DOM_OPT_FLIGHT_VIRT_MIN = (function() {
    const v = Math.floor(Number(DOM_OPT_CFG.flightListVirtualMinRows));
    return (isFinite(v) && v >= 8) ? v : 48;
  })();
  const DOM_OPT_FLIGHT_VIRT_OVERSCAN = (function() {
    const v = Math.floor(Number(DOM_OPT_CFG.flightListVirtualOverscan));
    return (isFinite(v) && v >= 0) ? v : 8;
  })();
  const DOM_OPT_FLIGHT_VIRT_ROW_H = (function() {
    const v = Number(DOM_OPT_CFG.flightListVirtualRowHeightPx);
    return (isFinite(v) && v >= 18) ? v : 28;
  })();
  const FLIGHT_SCHED_PAGE_SIZE = (function() {
    const v = Math.floor(Number(DOM_OPT_CFG.flightSchedulePageSize));
    if (!isFinite(v) || v < 0) return 20;
    return v;
  })();
  const GANTT_LEGEND_MAX_INTERVALS = (function() {
    const v = Math.floor(Number(DOM_OPT_CFG.ganttLegendMaxIntervals));
    if (!isFinite(v) || v < 1) return 100;
    return v;
  })();
  const KPI_ROLLING_TABLE_VISIBLE_ROWS = (function() {
    const v = Math.floor(Number(DOM_OPT_CFG.kpiRollingTableVisibleRows));
    if (!isFinite(v) || v < 1) return 24;
    return v;
  })();
  function _taNum(k, def) {
    const v = Number(TIME_AXIS_CFG[k]);
    return (isFinite(v) && v >= 0) ? v : def;
  }
  const GANTT_PAD_MIN = _taNum('apronGanttPadMin', 20);
  const RWY_SEP_TIMELINE_PAD_MIN = _taNum('runwaySepTimelinePadMin', 10);
  const TICK_STEP_SPAN_LE60 = _taNum('tickStepWhenSpanLe60Min', 10);
  const TICK_STEP_SPAN_LE240 = _taNum('tickStepWhenSpanLe240Min', 30);
  const TICK_STEP_ELSE = _taNum('tickStepElseMin', 60);
  const MAX_TICKS_SHOWN = (function() {
    const v = Math.floor(Number(TIME_AXIS_CFG.maxTicksShown));
    return (isFinite(v) && v >= 2) ? v : 6;
  })();
  const PATH_SEARCH_CFG = _algoTier.pathSearch || {};
  const TAXIWAY_HEURISTIC_COST = (function() {
    const v = Number(PATH_SEARCH_CFG.taxiwayHeuristicCost);
    return (isFinite(v) && v > 0) ? v : 200;
  })();
  const _ix = _layoutTier.interaction || {};
  function _interactionConfigNum(k, def) {
    const v = Number(_ix[k]);
    return (isFinite(v) && v >= 0) ? v : def;
  }
  function _ixBool(k, def) {
    const v = _ix[k];
    if (typeof v === 'boolean') return v;
    if (typeof v === 'number') return v !== 0;
    if (typeof v === 'string') {
      const s = v.trim().toLowerCase();
      if (s === 'true' || s === '1' || s === 'yes' || s === 'on') return true;
      if (s === 'false' || s === '0' || s === 'no' || s === 'off') return false;
    }
    return !!def;
  }
  const LAYOUT_VERTEX_DOT_SCALE = Math.max(0.25, Math.min(1.5, _interactionConfigNum('layoutVertexDotScale', 0.7)));
  const LAYOUT_SELECTED_VERTEX_RADIUS_FACTOR = Math.max(0.25, Math.min(1.5, _interactionConfigNum('layoutSelectedVertexRadiusFactor', 0.7)));
  const GRID_VISIBLE_DEFAULT = _ixBool('showGridDefault', true);
  const IMAGE_VISIBLE_DEFAULT = _ixBool('showImageDefault', true);
  const ROAD_WIDTH_VISIBLE_DEFAULT = _ixBool('showRoadWidthDefault', true);
  const DEFAULT_LAYERS = {
    grid: GRID_VISIBLE_DEFAULT,
    image: IMAGE_VISIBLE_DEFAULT,
    pathLines: true,
    pathFill: ROAD_WIDTH_VISIBLE_DEFAULT,
    standLines: true,
    standFill: true,
    islandAreaLines: true,
    islandAreaFill: ROAD_WIDTH_VISIBLE_DEFAULT,
    buildingLines: true,
    buildingFill: true,
    textRuler: false,
    dummyFlight: false,
    junction: true
  };
  const RW_EXIT_ALLOWED_DEFAULT = normalizeAllowedRunwayDirections(_dc.rwExitAllowedDefaultRaw);
  function layoutPathVertexRadiusPx(vertexSelected, pathSelected) {
    if (vertexSelected) return 6 * LAYOUT_VERTEX_DOT_SCALE * LAYOUT_SELECTED_VERTEX_RADIUS_FACTOR;
    if (pathSelected) return 5 * LAYOUT_VERTEX_DOT_SCALE * LAYOUT_SELECTED_VERTEX_RADIUS_FACTOR;
    return 4 * LAYOUT_VERTEX_DOT_SCALE;
  }
  function layoutTerminalVertexRadiusPx(vertexSelected) {
    return vertexSelected ? 5.5 * LAYOUT_VERTEX_DOT_SCALE * LAYOUT_SELECTED_VERTEX_RADIUS_FACTOR : 4 * LAYOUT_VERTEX_DOT_SCALE;
  }
  const _dragThreshPx = _interactionConfigNum('dragThresholdPx', 4);
  const DRAG_THRESH = _dragThreshPx > 0 ? Math.max(1, _dragThreshPx) : 4;
  const FREE_DRAW_STEP_CELL = Math.max(0.001, _interactionConfigNum('freeDrawStepCell', 0.05));
  const GRID_SNAP_STEP_CELL = Math.max(0.001, _interactionConfigNum('gridSnapStepCell', 0.5));
  const INSERT_VERTEX_HIT_CF = _interactionConfigNum('insertVertexHitCellFactor', 0.9);
  const CANVAS_MIN_ZOOM = Math.max(0.01, _interactionConfigNum('canvasMinZoom', 0.05));
  const CANVAS_MAX_ZOOM = Math.max(CANVAS_MIN_ZOOM, _interactionConfigNum('canvasMaxZoom', 10));
  const HIT_TERM_VTX_CF = _interactionConfigNum('hitTerminalVertexCellFactor', 0.6) * LAYOUT_VERTEX_DOT_SCALE;
  const HIT_TW_VTX_CF = _interactionConfigNum('hitTaxiwayVertexCellFactor', 0.6) * LAYOUT_VERTEX_DOT_SCALE;
  const HIT_TW_SEG_CF = _interactionConfigNum('hitTaxiwayAlongCellFactor', 0.8);
  const HIT_PBB_END_CF = _interactionConfigNum('hitPbbEndCellFactor', 0.8);
