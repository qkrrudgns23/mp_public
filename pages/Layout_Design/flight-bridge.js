  function pbbStandOverlapsTerminal(pbb) {
    const corners = getPBBStandCorners(pbb);
    for (let t = 0; t < state.terminals.length; t++) {
      const term = state.terminals[t];
      if (!term.closed || term.vertices.length < 3) continue;
      const termPix = term.vertices.map(v => cellToPixel(v.col, v.row));
      for (let k = 0; k < 4; k++) {
        if (pointInPolygonXY(corners[k], termPix)) return true;
      }
      for (let k = 0; k < termPix.length; k++) {
        if (pointInPolygonXY(termPix[k], corners)) return true;
      }
    }
    return false;
  }
  function pbbStandOverlapsExisting(pbb, excludeId) {
    if (pbbStandOverlapsTerminal(pbb)) return true;
    const cat = pbb.category || 'C';
    const center = getStandConnectionPx(pbb);
    const angle = getPBBStandAngle(pbb);
    if (standGapLineHitsExistingOuterContours(center, angle, cat)) return true;
    return false;
  }
  function pbbStandOuterContoursOverlapExisting(pbb, excludeId) {
    const corners = getPBBStandCorners(pbb);
    for (let i = 0; i < state.pbbStands.length; i++) {
      const other = state.pbbStands[i];
      if (!other) continue;
      if (excludeId && other.id === excludeId) continue;
      if (rotatedRectsOverlap(corners, getPBBStandCorners(other))) return true;
    }
    for (let i = 0; i < state.remoteStands.length; i++) {
      const st = state.remoteStands[i];
      if (!st) continue;
      if (rotatedRectsOverlap(corners, getRemoteStandCorners(st))) return true;
    }
    const temps = state.tempStands || [];
    for (let i = 0; i < temps.length; i++) {
      const st = temps[i];
      if (!st) continue;
      if (rotatedRectsOverlap(corners, getRemoteStandCorners(st))) return true;
    }
    return false;
  }
  function tryPlacePbbAt(wx, wy) {
    let bestEdge = null, bestD2 = Infinity;
    state.terminals.forEach(t => {
      if (!t.closed || t.vertices.length < 2) return;
      let cx = 0, cy = 0;
      t.vertices.forEach(v => { const [px, py] = cellToPixel(v.col, v.row); cx += px; cy += py; });
      cx /= t.vertices.length || 1; cy /= t.vertices.length || 1;
      for (let i = 0; i < t.vertices.length; i++) {
        const v1 = t.vertices[i], v2 = t.vertices[(i + 1) % t.vertices.length];
        const p1 = cellToPixel(v1.col, v1.row), p2 = cellToPixel(v2.col, v2.row);
        const near = closestPointOnSegment(p1, p2, [wx, wy]);
        if (near) {
          const d2 = dist2(near, [wx, wy]);
          if (d2 < bestD2) { bestD2 = d2; bestEdge = { near, p1, p2, col: v1.col, row: v1.row, cx, cy }; }
        }
      }
    });
    const maxD2 = (CELL_SIZE * TRY_PBB_MAX_EDGE_CF) ** 2;
    if (!bestEdge || bestD2 >= maxD2) return false;
    const [ex, ey] = bestEdge.near, [x1, y1] = bestEdge.p1, [x2, y2] = bestEdge.p2;
    let nx = -(y2 - y1), ny = x2 - x1;
    const len = Math.hypot(nx, ny) || 1; nx /= len; ny /= len;
    const inX = bestEdge.cx - ex, inY = bestEdge.cy - ey;
    if (nx * inX + ny * inY > 0) { nx *= -1; ny *= -1; }
    const categoryMode = normalizeStandCategoryMode(document.getElementById('standCategoryMode') ? document.getElementById('standCategoryMode').value : (_pbbTier.defaultCategoryMode || 'icao'), 'icao');
    const category = document.getElementById('standCategory').value || 'C';
    const minLen = getStandDepthMeters(category) / 2 + 3;
    const lenMeters = Number(document.getElementById('pbbLength').value || 15);
    const armLen = Math.max(isFinite(lenMeters) && lenMeters > 0 ? lenMeters : 15, minLen);
    const standAngleDeg = normalizeAngleDeg(Math.atan2(ny, nx) * 180 / Math.PI);
    const bwEl = document.getElementById('pbbBoardingWidth');
    const bhEl = document.getElementById('pbbBoardingHeight');
    const boardingW = Math.max(0.5, Number(bwEl && bwEl.value) || 5);
    const boardingH = Math.max(0.5, Number(bhEl && bhEl.value) || 15);
    const wallX = ex, wallY = ey;
    const bxOut = wallX + nx * boardingH, byOut = wallY + ny * boardingH;
    const cfgRow = standConfigRowForIcaoCat(category);
    const noseClear = cfgRow ? Number(cfgRow.nose_clear) : NaN;
    const offM = (Number.isFinite(noseClear) && noseClear > 0)
      ? noseClear
      : PBB_STAND_CENTER_OFFSET_FROM_TERMINAL_WALL_M;
    const newPbb = {
      x1: wallX, y1: wallY, x2: bxOut, y2: byOut, category,
      angleDeg: standAngleDeg,
      apronSiteX: wallX + nx * offM,
      apronSiteY: wallY + ny * offM,
      terminalContactSetbackM: offM,
      boardingWidthM: boardingW,
      boardingHeightM: boardingH
    };
    if (pbbStandOverlapsExisting(newPbb)) return false;
    const pbbNameCandidate = document.getElementById('standName').value.trim() || getDefaultPbbStandName();
    if (findDuplicateLayoutName('pbb', null, pbbNameCandidate)) {
      alertDuplicateLayoutName();
      return false;
    }
    pushUndo();
    state.pbbStands.push(normalizePbbStandObject({
      id: id(),
      name: pbbNameCandidate,
      x1: wallX, y1: wallY, x2: bxOut, y2: byOut,
      category: newPbb.category,
      terminalContactSetbackM: offM,
      categoryMode: categoryMode,
      allowedAircraftTypes: readCheckedDataItemIds('standAircraftAccess', '.aircraft-type-check'),
      pbbCount: Math.max(1, Math.min(8, parseInt(document.getElementById('pbbBridgeCount') ? document.getElementById('pbbBridgeCount').value : (_pbbTier.defaultBridgeCount || 1), 10) || 1)),
      angleDeg: standAngleDeg,
      apronSiteX: newPbb.apronSiteX,
      apronSiteY: newPbb.apronSiteY,
      boardingWidthM: boardingW,
      boardingHeightM: boardingH,
      pbbArmLenM: armLen,
      edgeCol: bestEdge.col,
      edgeRow: bestEdge.row
    }));
    return true;
  }
  function tryPlaceRemoteAt(wx, wy) {
    if (!isFinite(wx) || !isFinite(wy)) return false;
    const maxX = GRID_COLS * CELL_SIZE, maxY = GRID_ROWS * CELL_SIZE;
    if (wx < 0 || wy < 0 || wx > maxX || wy > maxY) return false;
    const categoryMode = normalizeStandCategoryMode(document.getElementById('remoteCategoryMode') ? document.getElementById('remoteCategoryMode').value : (_remoteTier.defaultCategoryMode || 'icao'), 'icao');
    const category = document.getElementById('remoteCategory').value || 'C';
    const angleInput = document.getElementById('remoteAngle');
    const angleDeg = normalizeAngleDeg(angleInput ? angleInput.value : 0);
    const candidate = { x: Number(wx), y: Number(wy), category, angleDeg };
    const candCorners = getRemoteStandCorners(candidate);
    for (let i = 0; i < state.remoteStands.length; i++) {
      const o = state.remoteStands[i];
      if (standFootprintsTooClose(candCorners, category, getRemoteStandCorners(o), o.category || 'C')) return false;
    }
    for (let i = 0; i < state.pbbStands.length; i++) {
      const o = state.pbbStands[i];
      if (standFootprintsTooClose(candCorners, category, getPBBStandCorners(o), o.category || 'C')) return false;
    }
    for (let i = 0; i < (state.tempStands || []).length; i++) {
      const o = state.tempStands[i];
      if (standFootprintsTooClose(candCorners, category, getRemoteStandCorners(o), o.category || 'C')) return false;
    }
    if (standGapLineHitsExistingOuterContours([Number(wx), Number(wy)], angleDeg * Math.PI / 180, category)) return false;
    const baseName = (document.getElementById('remoteName') && document.getElementById('remoteName').value.trim()) || getDefaultRemoteStandName();
    if (findDuplicateLayoutName('remote', null, baseName)) {
      alertDuplicateLayoutName();
      return false;
    }
    pushUndo();
    state.remoteStands.push(normalizeRemoteStandObject({
      id: id(),
      x: Number(wx),
      y: Number(wy),
      category,
      name: baseName,
      angleDeg,
      categoryMode: categoryMode,
      allowedAircraftTypes: readCheckedDataItemIds('remoteAircraftAccess', '.aircraft-type-check'),
      allowedTerminals: Array.from((document.getElementById('remoteTerminalAccess') || document).querySelectorAll('.remote-term-check')).filter(function(ch) { return ch.checked; }).map(function(ch) { return String(ch.getAttribute('data-item-id') || '').trim(); }).filter(Boolean)
    }));
    return true;
  }
  function tryPlaceTempStandAt(wx, wy) {
    const snap = snapTempStandOnTaxiwayCenterlines(wx, wy);
    if (!snap) return false;
    const sx = snap.x, sy = snap.y;
    const categoryMode = normalizeStandCategoryMode(document.getElementById('tempStandCategoryMode') ? document.getElementById('tempStandCategoryMode').value : (_remoteTier.defaultCategoryMode || 'icao'), 'icao');
    const category = document.getElementById('tempStandCategory') ? document.getElementById('tempStandCategory').value || 'C' : 'C';
    const angleInput = document.getElementById('tempStandAngle');
    const angleDeg = normalizeAngleDeg(angleInput ? angleInput.value : 0);
    const candidate = { x: Number(sx), y: Number(sy), category, angleDeg };
    const candCorners = getRemoteStandCorners(candidate);
    for (let i = 0; i < (state.tempStands || []).length; i++) {
      const o = state.tempStands[i];
      if (standFootprintsTooClose(candCorners, category, getRemoteStandCorners(o), o.category || 'C')) return false;
    }
    for (let i = 0; i < state.remoteStands.length; i++) {
      const o = state.remoteStands[i];
      if (standFootprintsTooClose(candCorners, category, getRemoteStandCorners(o), o.category || 'C')) return false;
    }
    for (let i = 0; i < state.pbbStands.length; i++) {
      const o = state.pbbStands[i];
      if (standFootprintsTooClose(candCorners, category, getPBBStandCorners(o), o.category || 'C')) return false;
    }
    if (standGapLineHitsExistingOuterContours([Number(sx), Number(sy)], angleDeg * Math.PI / 180, category)) return false;
    const baseName = (document.getElementById('tempStandName') && document.getElementById('tempStandName').value.trim()) || getDefaultTempStandName();
    if (findDuplicateLayoutName('tempStand', null, baseName)) {
      alertDuplicateLayoutName();
      return false;
    }
    pushUndo();
    state.tempStands.push(normalizeTempStandObject({
      id: id(),
      x: Number(sx),
      y: Number(sy),
      junctionX: Number(sx),
      junctionY: Number(sy),
      category,
      name: baseName,
      angleDeg,
      categoryMode: categoryMode,
      allowedAircraftTypes: readCheckedDataItemIds('tempStandAircraftAccess', '.aircraft-type-check'),
      allowedTerminals: Array.from((document.getElementById('tempStandTerminalAccess') || document).querySelectorAll('.remote-term-check')).filter(function(ch) { return ch.checked; }).map(function(ch) { return String(ch.getAttribute('data-item-id') || '').trim(); }).filter(Boolean)
    }));
    return true;
  }
  function taxiwayOverlapsAnyTerminal(tw) {
    if (!tw || !tw.vertices || tw.vertices.length < 2) return false;
    const vertsPix = tw.vertices.map(v => cellToPixel(v.col, v.row));
    for (let t = 0; t < state.terminals.length; t++) {
      const term = state.terminals[t];
      if (!term.closed || term.vertices.length < 3) continue;
      const termPix = term.vertices.map(v => cellToPixel(v.col, v.row));
      for (let i = 0; i < vertsPix.length; i++) {
        if (pointInPolygonXY(vertsPix[i], termPix)) return true;
      }
      for (let i = 0; i < vertsPix.length - 1; i++) {
        const a1 = vertsPix[i], a2 = vertsPix[i+1];
        for (let j = 0; j < termPix.length; j++) {
          const b1 = termPix[j], b2 = termPix[(j+1) % termPix.length];
          if (segIntersect(a1, a2, b1, b2)) return true;
        }
      }
    }
    return false;


  }
  function terminalOverlapsAnyTaxiway(term) {
    if (!term || !term.vertices || term.vertices.length < 3) return false;
    const termPix = term.vertices.map(v => cellToPixel(v.col, v.row));
    if (!state.taxiways || !state.taxiways.length) return false;
    for (let i = 0; i < state.taxiways.length; i++) {
      const tw = state.taxiways[i];
      if (!tw.vertices || tw.vertices.length < 2) continue;
      const vertsPix = tw.vertices.map(v => cellToPixel(v.col, v.row));
      for (let k = 0; k < vertsPix.length; k++) {
        if (pointInPolygonXY(vertsPix[k], termPix)) return true;
      }
      for (let a = 0; a < vertsPix.length - 1; a++) {
        const a1 = vertsPix[a], a2 = vertsPix[a+1];
        for (let b = 0; b < termPix.length; b++) {
          const b1 = termPix[b], b2 = termPix[(b+1) % termPix.length];
          if (segIntersect(a1, a2, b1, b2)) return true;
        }
      }
    }
    return false;
  }
  function makeUniqueNamedCopy(list, _prop) {
    return (list || []).map(function(obj) {
      return Object.assign({}, obj);
    });
  }

  function _persistCellSizePx() {
    return (typeof CELL_SIZE === 'number' && CELL_SIZE > 0) ? CELL_SIZE : 20;
  }
  function persistVerticesCellsToXY(vertices) {
    const cs = _persistCellSizePx();
    if (!Array.isArray(vertices)) return [];
    return vertices.map(function(v) {
      if (!v || typeof v !== 'object') return { x: 0, y: 0 };
      const c = Number(v.col), r = Number(v.row);
      return { x: (isFinite(c) ? c : 0) * cs, y: (isFinite(r) ? r : 0) * cs };
    });
  }
  function persistPointCellToXY(pt) {
    if (!pt || typeof pt !== 'object') return null;
    const xRaw = Number(pt.x), yRaw = Number(pt.y);
    if (isFinite(xRaw) && isFinite(yRaw)) return { x: xRaw, y: yRaw };
    const cs = _persistCellSizePx();
    const c = Number(pt.col), r = Number(pt.row);
    return { x: (isFinite(c) ? c : 0) * cs, y: (isFinite(r) ? r : 0) * cs };
  }

  function _polylineLengthPxForLineup(pts) {
    if (!pts || pts.length < 2) return 0;
    let s = 0;
    for (let i = 0; i < pts.length - 1; i++) {
      const p1 = pts[i], p2 = pts[i + 1];
      s += Math.hypot(p2[0] - p1[0], p2[1] - p1[1]);
    }
    return s;
  }
  function _pointOnPolylineAtDistPxForLineup(pts, distPx) {
    if (!pts || pts.length < 2) return null;
    const total = _polylineLengthPxForLineup(pts);
    const d = Math.max(0, Math.min(typeof distPx === 'number' ? distPx : 0, total));
    let acc = 0;
    for (let i = 0; i < pts.length - 1; i++) {
      const p1 = pts[i], p2 = pts[i + 1];
      const segLen = Math.hypot(p2[0] - p1[0], p2[1] - p1[1]);
      if (!(segLen > 1e-6)) continue;
      if (acc + segLen >= d - 1e-6) {
        const t = Math.max(0, Math.min(1, (d - acc) / segLen));
        return [p1[0] + (p2[0] - p1[0]) * t, p1[1] + (p2[1] - p1[1]) * t];
      }
      acc += segLen;
    }
    const last = pts[pts.length - 1];
    return [last[0], last[1]];
  }
  /** Ordered runway polyline in layout px (matches getRunwayPath / departure graphPath). */
  function _persistRunwayPolylinePtsPx(tw) {
    if (!tw || tw.pathType !== 'runway' || !tw.vertices || tw.vertices.length < 2) return null;
    return tw.vertices.map(function(v) { return cellToPixel(v.col, v.row); });
  }

  function serializeTaxiwayWithEndpoints(tw) {
    const copy = Object.assign({}, tw);
    if (Array.isArray(tw.vertices)) {
      copy.vertices = persistVerticesCellsToXY(tw.vertices.slice());
    }
    delete copy.start_point;
    delete copy.end_point;
    if (typeof tw.avgMoveVelocity === 'number' && isFinite(tw.avgMoveVelocity) && tw.avgMoveVelocity > 0) {
      copy.avgMoveVelocity = tw.avgMoveVelocity;
    }
    if (tw.pathType === 'runway' && typeof tw.minArrVelocity === 'number' && isFinite(tw.minArrVelocity) && tw.minArrVelocity > 0) {
      copy.minArrVelocity = Math.max(1, Math.min(150, tw.minArrVelocity));
    }
    if (tw.pathType === 'runway') {
      const lCw = getRunwayLineupDistMByDirection(tw, 'clockwise');
      const lCcw = getRunwayLineupDistMByDirection(tw, 'counter_clockwise');
      copy.lineupDistM_CW = lCw;
      copy.lineupDistM_CCW = lCcw;
      copy.lineupDistM = getEffectiveRunwayLineupDistM(tw);
      if (typeof tw.startDisplacedThresholdM === 'number' && isFinite(tw.startDisplacedThresholdM) && tw.startDisplacedThresholdM >= 0) copy.startDisplacedThresholdM = tw.startDisplacedThresholdM;
      else delete copy.startDisplacedThresholdM;
      if (typeof tw.startBlastPadM === 'number' && isFinite(tw.startBlastPadM) && tw.startBlastPadM >= 0) copy.startBlastPadM = tw.startBlastPadM;
      else delete copy.startBlastPadM;
      if (typeof tw.endDisplacedThresholdM === 'number' && isFinite(tw.endDisplacedThresholdM) && tw.endDisplacedThresholdM >= 0) copy.endDisplacedThresholdM = tw.endDisplacedThresholdM;
      else delete copy.endDisplacedThresholdM;
      if (typeof tw.endBlastPadM === 'number' && isFinite(tw.endBlastPadM) && tw.endBlastPadM >= 0) copy.endBlastPadM = tw.endBlastPadM;
      else delete copy.endBlastPadM;
      const rwPts = _persistRunwayPolylinePtsPx(tw);
      if (rwPts && rwPts.length >= 2) {
        const lenPx = _polylineLengthPxForLineup(rwPts);
        const dPx = getEffectiveRunwayLineupDistFromStartM(tw, lenPx);
        const lp = _pointOnPolylineAtDistPxForLineup(rwPts, dPx);
        if (lp) copy.lineup_point = { x: lp[0], y: lp[1] };
        else delete copy.lineup_point;
      } else {
        delete copy.lineup_point;
      }
      delete copy.dep_point;
      delete copy.depPointPos;
    }
    if (tw.pathType === 'runway' && tw.rwySepConfig) copy.rwySepConfig = tw.rwySepConfig;
    else delete copy.rwySepConfig;
    return copy;
  }
  function partitionTaxiwaysForPersist(list) {
    const runwayPaths = [];
    const runwayTaxiways = [];
    const taxiways = [];
    (list || []).forEach(function(tw) {
      const ser = serializeTaxiwayWithEndpoints(tw);
      const pt = tw.pathType || 'taxiway';
      delete ser.pathType;
      if (pt === 'runway') runwayPaths.push(ser);
      else if (pt === 'runway_exit') runwayTaxiways.push(ser);
      else {
        taxiways.push(ser);
      }
    });
    return { runwayPaths: runwayPaths, runwayTaxiways: runwayTaxiways, taxiways: taxiways };
  }
  function serializeCurrentLayout() {
    function pathJunctionsToNetworkJunctions(pts) {
      const out = [];
      (pts || []).forEach(function(p) {
        if (!p) return;
        if (Array.isArray(p) && p.length >= 2) {
          out.push({ x: p[0], y: p[1] });
        } else if (typeof p.x === 'number' && typeof p.y === 'number') {
          out.push({ x: p.x, y: p.y });
        }
      });
      return out;
    }
    let networkJunctions = pathJunctionsToNetworkJunctions(state.pathGraphJunctions);
    if (!networkJunctions.length && typeof buildPathGraph === 'function') {
      try {
        let gj = null;
        const sig = computeTaxiwaysGraphSig();
        if (state.pathGraphCacheValid && state.pathGraphCache && state.pathGraphCacheSig === sig) {
          gj = state.pathGraphCache;
        } else if (!PATH_GRAPH_SYNC_ONLY_ON_EXPLICIT_ACTION) {
          gj = buildPathGraph(null);
        }
        if (gj) {
          const cj = (gj && (gj.connectedJunctions || gj.junctions)) || [];
          networkJunctions = pathJunctionsToNetworkJunctions(cj);
        }
      } catch (e) { /* ignore */ }
    }
    let edgeExport = [];
    if (typeof rebuildDerivedGraphEdges === 'function') {
      try {
        rebuildDerivedGraphEdges();
        edgeExport = (state.derivedGraphEdges || []).map(function(ed) {
          return { id: ed.id, label: ed.label, name: ed.name, fromIdx: ed.fromIdx, toIdx: ed.toIdx };
        });
      } catch (e2) { edgeExport = []; }
    }
    return {
      grid: {
        cols: GRID_COLS,
        rows: GRID_ROWS,
        cellSize: CELL_SIZE,
        showGrid: !!state.showGrid,
        showImage: !!state.showImage,
        showRoadWidth: !!state.showRoadWidth,
        showLayoutMarkers: !!state.showLayoutMarkers,
        layers: Object.assign({}, state.layers),
        layoutImageOverlay: state.layoutImageOverlay ? Object.assign({}, state.layoutImageOverlay) : null
      },
      networkJunctions: networkJunctions,
      Edge: edgeExport,
      terminals: makeUniqueNamedCopy(state.terminals, 'name').map(function(t) {
        const o = Object.assign({}, t);
        if (Array.isArray(o.vertices)) o.vertices = persistVerticesCellsToXY(o.vertices);
        return o;
      }),
      pbbStands: makeUniqueNamedCopy(state.pbbStands, 'name'),
      remoteStands: state.remoteStands.slice(),
      tempStands: (state.tempStands || []).slice(),
      holdingPoints: (state.holdingPoints || []).slice(),
      ...(function() {
        const p = partitionTaxiwaysForPersist(state.taxiways);
        return { runwayPaths: p.runwayPaths, runwayTaxiways: p.runwayTaxiways, taxiways: p.taxiways };
      })(),
      apronLinks: (state.apronLinks || []).map(function(lk) {
        const o = Object.assign({}, lk);
        if (Array.isArray(o.midVertices)) o.midVertices = persistVerticesCellsToXY(o.midVertices);
        return o;
      }),
      directionModes: state.directionModes.slice(),
      flights: state.flights.map(function(f) {
        const copy = {};
        const simFlightKeys = [
          'id',
          'reg',
          'airlineCode',
          'flightNumber',
          'aircraftType',
          'code',
