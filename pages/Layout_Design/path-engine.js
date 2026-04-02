    const dax = ax2-ax1, day = ay2-ay1, dbx = bx2-bx1, dby = by2-by1;
    const den = dax*dby - day*dbx;
    if (Math.abs(den) < 1e-10) return false;
    const t = ((bx1-ax1)*dby - (by1-ay1)*dbx) / den;
    const s = ((bx1-ax1)*day - (by1-ay1)*dax) / den;
    return t >= 0 && t <= 1 && s >= 0 && s <= 1;
  }
  function rotatedRectsOverlap(cornersA, cornersB) {
    for (let i = 0; i < 4; i++) if (pointInPolygonXY(cornersA[i], cornersB)) return true;
    for (let i = 0; i < 4; i++) if (pointInPolygonXY(cornersB[i], cornersA)) return true;
    for (let i = 0; i < 4; i++) {
      const a1 = cornersA[i], a2 = cornersA[(i+1)%4];
      for (let j = 0; j < 4; j++) {
        if (segIntersect(a1, a2, cornersB[j], cornersB[(j+1)%4])) return true;
      }
    }
    return false;
  }
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
    const corners = getPBBStandCorners(pbb);
    for (let i = 0; i < state.pbbStands.length; i++) {
      const other = state.pbbStands[i];
      if (excludeId && other.id === excludeId) continue;
      if (rotatedRectsOverlap(corners, getPBBStandCorners(other))) return true;
    }
    for (let i = 0; i < state.remoteStands.length; i++) {
      const st = state.remoteStands[i];
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
    const toClickX = wx - ex, toClickY = wy - ey;
    if (nx * toClickX + ny * toClickY < 0) { nx *= -1; ny *= -1; }
    const categoryMode = normalizeStandCategoryMode(document.getElementById('standCategoryMode') ? document.getElementById('standCategoryMode').value : (_pbbTier.defaultCategoryMode || 'icao'), 'icao');
    const category = document.getElementById('standCategory').value || 'C';
    const standSize = getStandSizeMeters(category);
    const minLen = standSize / 2 + 3;
    const lenMeters = Number(document.getElementById('pbbLength').value || 15);
    const lenPx = Math.max(isFinite(lenMeters) && lenMeters > 0 ? lenMeters : 15, minLen);
    const newPbb = { x1: ex, y1: ey, x2: ex + nx * lenPx, y2: ey + ny * lenPx, category };
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
      x1: ex, y1: ey, x2: ex + nx * lenPx, y2: ey + ny * lenPx,
      category: newPbb.category,
      categoryMode: categoryMode,
      allowedAircraftTypes: readCheckedDataItemIds('standAircraftAccess', '.aircraft-type-check'),
      pbbCount: Math.max(1, Math.min(8, parseInt(document.getElementById('pbbBridgeCount') ? document.getElementById('pbbBridgeCount').value : (_pbbTier.defaultBridgeCount || 1), 10) || 1)),
      angleDeg: normalizeAngleDeg(Math.atan2(ny, nx) * 180 / Math.PI),
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
      if (rotatedRectsOverlap(candCorners, getRemoteStandCorners(state.remoteStands[i]))) return false;
    }
    for (let i = 0; i < state.pbbStands.length; i++) {
      if (rotatedRectsOverlap(candCorners, getPBBStandCorners(state.pbbStands[i]))) return false;
    }
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
    const cs = _persistCellSizePx();
    const c = Number(pt.col), r = Number(pt.row);
    return { x: (isFinite(c) ? c : 0) * cs, y: (isFinite(r) ? r : 0) * cs };
  }

  function serializeTaxiwayWithEndpoints(tw) {
    const copy = Object.assign({}, tw);
    const dir = getTaxiwayDirection(tw);
    if (dir === 'both') {
      copy.start_point = null;
      copy.end_point = null;
    } else {
      if (tw.vertices && tw.vertices.length >= 2) {
        const first = tw.vertices[0];
        const last = tw.vertices[tw.vertices.length - 1];
        if (dir === 'clockwise') {
          copy.start_point = persistPointCellToXY({ col: first.col, row: first.row });
          copy.end_point = persistPointCellToXY({ col: last.col, row: last.row });
        } else {
          copy.start_point = persistPointCellToXY({ col: last.col, row: last.row });
          copy.end_point = persistPointCellToXY({ col: first.col, row: first.row });
        }
      } else {
        copy.start_point = null;
        copy.end_point = null;
      }
    }
    if (typeof tw.avgMoveVelocity === 'number' && isFinite(tw.avgMoveVelocity) && tw.avgMoveVelocity > 0) {
      copy.avgMoveVelocity = tw.avgMoveVelocity;
    }
    if (tw.pathType === 'runway' && typeof tw.minArrVelocity === 'number' && isFinite(tw.minArrVelocity) && tw.minArrVelocity > 0) {
      copy.minArrVelocity = Math.max(1, Math.min(150, tw.minArrVelocity));
    }
    if (tw.pathType === 'runway') {
      if (typeof tw.lineupDistM === 'number' && isFinite(tw.lineupDistM) && tw.lineupDistM >= 0) copy.lineupDistM = tw.lineupDistM;
      else delete copy.lineupDistM;
      if (typeof tw.startDisplacedThresholdM === 'number' && isFinite(tw.startDisplacedThresholdM) && tw.startDisplacedThresholdM >= 0) copy.startDisplacedThresholdM = tw.startDisplacedThresholdM;
      else delete copy.startDisplacedThresholdM;
      if (typeof tw.startBlastPadM === 'number' && isFinite(tw.startBlastPadM) && tw.startBlastPadM >= 0) copy.startBlastPadM = tw.startBlastPadM;
      else delete copy.startBlastPadM;
      if (typeof tw.endDisplacedThresholdM === 'number' && isFinite(tw.endDisplacedThresholdM) && tw.endDisplacedThresholdM >= 0) copy.endDisplacedThresholdM = tw.endDisplacedThresholdM;
      else delete copy.endDisplacedThresholdM;
      if (typeof tw.endBlastPadM === 'number' && isFinite(tw.endBlastPadM) && tw.endBlastPadM >= 0) copy.endBlastPadM = tw.endBlastPadM;
      else delete copy.endBlastPadM;
      delete copy.lineup_point;
      delete copy.dep_point;
      delete copy.depPointPos;
    }
    if (tw.pathType === 'runway' && tw.rwySepConfig) copy.rwySepConfig = tw.rwySepConfig;
    else delete copy.rwySepConfig;
    if (Array.isArray(tw.vertices)) copy.vertices = persistVerticesCellsToXY(tw.vertices);
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
      else taxiways.push(ser);
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
        const g = buildPathGraph(null);
        const cj = (g && (g.connectedJunctions || g.junctions)) || [];
        networkJunctions = pathJunctionsToNetworkJunctions(cj);
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
          'timeMin',
          'dwellMin',
          'minDwellMin',
          'noWayArr',
          'noWayDep',
          'eOverlapPushed',
          'arrRetFailed',
          'serviceDate',
          'sldtMin_orig',
          'sibtMin_orig',
          'sobtMin_orig',
          'stotMin_orig',
          'sldtMin_d',
          'sibtMin_d',
          'sobtMin_d',
          'stotMin_d',
          'arrRunwayDirUsed',
          'depRunwayDirUsed',
          'arrTdDistM',
          'arrVTdMs',
          'arrDecelMs2',
          'arrDep',
        ];
        simFlightKeys.forEach(function(k) {
          if (Object.prototype.hasOwnProperty.call(f, k) && f[k] !== undefined) {
            copy[k] = f[k];
          }
        });
        if (Array.isArray(f.edge_list) && f.edge_list.length) {
          copy.edge_list = f.edge_list.slice();
        }
        const t = f.token || {};
        const arrRwyId = f.arrRunwayId || t.arrRunwayId || t.runwayId || null;
        const apronId = (f.standId != null ? f.standId : (t.apronId != null ? t.apronId : null));
        const termId = f.terminalId || t.terminalId || null;
        const depRwyId = f.depRunwayId || t.depRunwayId || null;
        const exitTwId = (f.sampledArrRet != null && f.sampledArrRet !== '') ? f.sampledArrRet : (t.ExitTaxiwayId != null ? t.ExitTaxiwayId : null);
        copy.token = {
          arrRunwayId: arrRwyId,
          ExitTaxiwayId: exitTwId || null,
          apronId: apronId || null,
          terminalId: termId || null,
          depRunwayId: depRwyId || null,
        };
        function _twNameById(id) {
          if (id == null || id === '') return null;
          const tw = (state.taxiways || []).find(function(x) { return x && x.id === id; });
          if (!tw) return String(id);
          const n = (tw.name && String(tw.name).trim()) || '';
          return n || String(tw.id || id);
        }
        function _standNameById(id) {
          if (id == null || id === '') return null;
          if (typeof findStandById === 'function') {
            const st = findStandById(id);
            if (!st) return String(id);
            const n = (st.name && String(st.name).trim()) || '';
            return n || String(st.id || id);
          }
          return String(id);
        }
        function _labelOrId(id, getLab) {
          if (id == null || id === '') return null;
          if (typeof getLab === 'function') {
            const lab = getLab(id);
            if (lab && lab !== '—') return lab;
          }
          return String(id);
        }
        copy.token_name = {
          arrRunwayId: _labelOrId(arrRwyId, typeof getRunwayDisplayLabelById === 'function' ? getRunwayDisplayLabelById : null),
          ExitTaxiwayId: exitTwId ? _twNameById(exitTwId) : null,
          apronId: apronId ? _standNameById(apronId) : null,
          terminalId: _labelOrId(termId, typeof getTerminalDisplayLabelById === 'function' ? getTerminalDisplayLabelById : null),
          depRunwayId: _labelOrId(depRwyId, typeof getRunwayDisplayLabelById === 'function' ? getRunwayDisplayLabelById : null),
        };
        return copy;
      }),
      simPathGraph: (typeof buildSimPathGraphExport === 'function' ? buildSimPathGraphExport() : null)
    };
  }
  function getExistingStandBounds() {
    const list = [];
    state.remoteStands.forEach(st => {
      const corners = getRemoteStandCorners(st);
      let left = corners[0][0], right = corners[0][0], top = corners[0][1], bottom = corners[0][1];
      for (let k = 1; k < 4; k++) {
        left = Math.min(left, corners[k][0]); right = Math.max(right, corners[k][0]);
        top = Math.min(top, corners[k][1]); bottom = Math.max(bottom, corners[k][1]);
      }
      list.push({ left, right, top, bottom });
    });
    state.pbbStands.forEach(pbb => {
      const corners = getPBBStandCorners(pbb);
      let left = corners[0][0], right = corners[0][0], top = corners[0][1], bottom = corners[0][1];
      for (let k = 1; k < 4; k++) {
        left = Math.min(left, corners[k][0]); right = Math.max(right, corners[k][0]);
        top = Math.min(top, corners[k][1]); bottom = Math.max(bottom, corners[k][1]);
      }
      list.push({ left, right, top, bottom });
    });
    return list;
  }
  function standOverlapsExisting(bounds) {
    const existing = getExistingStandBounds();
    for (let i = 0; i < existing.length; i++) if (rectsOverlap(bounds, existing[i])) return true;
    return false;
  }
  function dist2(a, b) { const dx = a[0]-b[0], dy = a[1]-b[1]; return dx*dx+dy*dy; }
  function _normalizeTimeToSeconds(value, unit, roundingMode) {
    const raw = Number(value || 0);
    const scaled = unit === 'minutes' ? raw * 60 : raw;
    const rounded = roundingMode === 'round' ? Math.round(scaled) : Math.floor(scaled);
    return Math.max(0, rounded);
  }
  function _splitTotalSeconds(totalSec) {
    const safeSec = Math.max(0, Math.floor(totalSec || 0));
    const h = Math.floor(safeSec / 3600);
    const m = Math.floor((safeSec % 3600) / 60);
    const s = safeSec % 60;
    return {
      h,
      m,
      s,
      hh: (h < 10 ? '0' : '') + h,
      mm: (m < 10 ? '0' : '') + m,
      ss: (s < 10 ? '0' : '') + s,
    };
  }
  function formatMinutesToHHMM(m) {
    const parts = _splitTotalSeconds(_normalizeTimeToSeconds(m, 'minutes', 'floor'));
    return parts.h + ':' + parts.mm;
  }
  function findNearestItem(candidates, getPoint, wx, wy, maxD2) {
    const click = [wx, wy];
    let best = null;
    let bestD2 = maxD2;
    for (let i = 0; i < candidates.length; i++) {
      const c = candidates[i];
      const pt = getPoint(c);
      if (!pt || pt.length < 2) continue;
      const d2 = dist2(pt, click);
      if (d2 < bestD2) {
        bestD2 = d2;
        best = c;
      }
    }
    return best;
  }
  function closestPointOnSegment(p1, p2, p) {
    const [x1,y1]=p1,[x2,y2]=p2,[px,py]=p;
    const dx=x2-x1,dy=y2-y1,len2=dx*dx+dy*dy;
    if (len2===0) return null;
    let t = ((px-x1)*dx+(py-y1)*dy)/len2;
    t = Math.max(0,Math.min(1,t));
    return [x1+t*dx,y1+t*dy];
  }
  function getClosestTerminalEdgePoint(wx, wy) {
    const click = [wx, wy];
    let best = null;
    let bestD2 = Infinity;
    (state.terminals || []).forEach(function(term) {
      if (!term || !term.closed || !Array.isArray(term.vertices) || term.vertices.length < 2) return;
      for (let i = 0; i < term.vertices.length; i++) {
        const v1 = term.vertices[i];
        const v2 = term.vertices[(i + 1) % term.vertices.length];
        const p1 = cellToPixel(v1.col, v1.row);
        const p2 = cellToPixel(v2.col, v2.row);
        const near = closestPointOnSegment(p1, p2, click);
        if (!near) continue;
        const d2 = dist2(near, click);
        if (d2 < bestD2) {
          bestD2 = d2;
          best = { point: near, term: term, edgeIndex: i };
        }
      }
    });
    return best;
  }

  function pointInPolygon(p, verts) {
    let inside = false;
    const n = verts.length;
    for (let i = 0, j = n - 1; i < n; j = i++) {
      const vi = cellToPixel(verts[i].col, verts[i].row);
      const vj = cellToPixel(verts[j].col, verts[j].row);
      if (((vi[1] > p[1]) !== (vj[1] > p[1])) && (p[0] < (vj[0]-vi[0])*(p[1]-vi[1])/(vj[1]-vi[1])+vi[0])) inside = !inside;
    }
    return inside;
  }

  function getApronLinkStandEndPx(lk) {
    if (!lk || !lk.pbbId) return null;
    const stand = findStandById(lk.pbbId);
    if (!stand) return null;
    return getStandConnectionPx(stand);
  }
  function getApronLinkPolylineWorldPts(lk) {
    if (!lk || lk.tx == null || lk.ty == null) return [];
    const a = getApronLinkStandEndPx(lk);
    if (!a) return [];
    const mids = (Array.isArray(lk.midVertices) ? lk.midVertices : []).map(function(v) {
      if (v && isFinite(Number(v.x)) && isFinite(Number(v.y))) return [Number(v.x), Number(v.y)];
      return cellToPixel(Number(v.col), Number(v.row));
    });
    const b = [Number(lk.tx), Number(lk.ty)];
    return [a].concat(mids).concat([b]);
  }
  function hitTestApronLink(wx, wy) {
    const click = [wx, wy];
    const hitD2 = (CELL_SIZE * HIT_TW_SEG_CF) ** 2;
    const list = state.apronLinks || [];
    for (let i = list.length - 1; i >= 0; i--) {
      const lk = list[i];
      const poly = getApronLinkPolylineWorldPts(lk);
      if (poly.length < 2) continue;
      for (let j = 0; j < poly.length - 1; j++) {
        const near = closestPointOnSegment(poly[j], poly[j + 1], click);
        if (!near) continue;
        if (dist2(near, click) < hitD2) return { type: 'apronLink', id: lk.id, obj: lk };
      }
    }
    return null;
  }

  function getDefaultHoldingPointLabel() {
    let maxN = 0;
    (state.holdingPoints || []).forEach(function(h) {
      const m = /^Position(\d+)$/i.exec(String(h && h.name || '').trim());
      if (m) maxN = Math.max(maxN, parseInt(m[1], 10));
    });
    return 'Position' + (maxN + 1);
  }
  function snapHoldingPointOnAllowedTaxiways(wx, wy) {
    const click = [wx, wy];
    const maxD2 = (CELL_SIZE * HIT_TW_SEG_CF) ** 2;
    let best = null;
    let bestD2 = maxD2;
    (state.taxiways || []).forEach(function(tw) {
      const pt = tw.pathType || 'taxiway';
      if (pt !== 'taxiway' && pt !== 'runway_exit') return;
      if (!tw.vertices || tw.vertices.length < 2) return;
      for (let i = 0; i < tw.vertices.length - 1; i++) {
        const [x1, y1] = cellToPixel(tw.vertices[i].col, tw.vertices[i].row);
        const [x2, y2] = cellToPixel(tw.vertices[i + 1].col, tw.vertices[i + 1].row);
        const near = closestPointOnSegment([x1, y1], [x2, y2], click);
        if (!near) continue;
        const d2 = dist2(near, click);
        if (d2 < bestD2) { bestD2 = d2; best = { x: near[0], y: near[1], pathType: pt }; }
      }
    });
    return best;
  }
  function hitTestHoldingPoint(wx, wy) {
    const r = c2dHoldingPointDiameterM() * 0.5;
    const rHit = r + Math.max(2, CELL_SIZE * 0.15);
    const r2 = rHit * rHit;
    const pts = state.holdingPoints || [];
    for (let i = pts.length - 1; i >= 0; i--) {
      const hp = pts[i];
      if (!hp || !isFinite(hp.x) || !isFinite(hp.y)) continue;
      const dx = wx - hp.x, dy = wy - hp.y;
      if (dx * dx + dy * dy <= r2) return { type: 'holdingPoint', id: hp.id, obj: hp };
    }
    return null;
  }
  function tryPlaceHoldingPointAt(x, y, pathType) {
    const hpKind = pathTypeToHpKind(pathType || 'taxiway');
    const nameInput = document.getElementById('holdingPointName');
    const manual = nameInput && nameInput.value && String(nameInput.value).trim();
    let baseName = manual ? String(nameInput.value).trim() : getDefaultHoldingPointLabel();
    if (findDuplicateLayoutName('holdingPoint', null, baseName)) { alertDuplicateLayoutName(); return false; }
    pushUndo();
    state.holdingPoints.push({ id: id(), name: baseName, x: x, y: y, hpKind: hpKind });
    return true;
  }

  function hitTest(wx, wy) {
    const click = [wx, wy];
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
      for (let i = state.taxiways.length - 1; i >= 0; i--) {
        const tw = state.taxiways[i];
        if (tw.vertices.length < 2) continue;
        const halfW = (tw.width != null ? tw.width : 23) / 2;
        const hitD2 = (CELL_SIZE * HIT_TW_SEG_CF + halfW) ** 2;
        for (let j = 0; j < tw.vertices.length - 1; j++) {
          const [x1, y1] = cellToPixel(tw.vertices[j].col, tw.vertices[j].row);
          const [x2, y2] = cellToPixel(tw.vertices[j + 1].col, tw.vertices[j + 1].row);
          const near = closestPointOnSegment([x1, y1], [x2, y2], click);
          if (near && dist2(near, click) < hitD2) return { type: 'taxiway', id: tw.id, obj: tw };
        }
      }
    }
    return null;
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
        const d2 = dist2([Number(pt.x) || 0, Number(pt.y) || 0], click);
        if (d2 < bestD2) {
          bestD2 = d2;
          best = { type: 'bridge', bridgeIndex: bridgeIdx, pointIndex: ptIdx };
        }
      });
    });
    const apronPt = getStandConnectionPx(pbb);
    const apronD2 = dist2(apronPt, click);
    if (apronD2 < bestD2) best = { type: 'apronSite' };
    return best;
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
  function insertSelectedVertexAt(wx, wy, snapToGrid) {
    if (!state.selectedObject || !state.selectedObject.obj) return false;
    const sel = state.selectedObject;
    if (sel.type === 'terminal') {
      const term = sel.obj;
      const hit = findInsertSegment(term.vertices, !!term.closed, wx, wy);
      if (!hit) return false;
      const pt = worldPointToCellPoint(hit.near[0], hit.near[1], snapToGrid);
      pushUndo();
      term.vertices.splice(hit.insertIndex, 0, pt);
      state.selectedVertex = { type: 'terminal', id: term.id, index: hit.insertIndex };
      updateObjectInfo();
      draw();
      return true;
    }
    if (sel.type === 'taxiway') {
      const tw = sel.obj;
      const hit = findInsertSegment(tw.vertices, false, wx, wy);
      if (!hit) return false;
      const pt = worldPointToCellPoint(hit.near[0], hit.near[1], snapToGrid);
      pushUndo();
      tw.vertices.splice(hit.insertIndex, 0, pt);
      if (typeof syncStartEndFromVertices === 'function') syncStartEndFromVertices(tw);
      state.selectedVertex = { type: 'taxiway', id: tw.id, index: hit.insertIndex };
      if (typeof redrawLayoutAfterEdit === 'function') redrawLayoutAfterEdit();
      else if (typeof updateAllFlightPaths === 'function') updateAllFlightPaths(); else draw();
      return true;
    }
    if (sel.type === 'apronLink') {
      const lk = sel.obj;
      const mids = (Array.isArray(lk.midVertices) ? lk.midVertices.slice() : []);
      const poly = [getApronLinkStandEndPx(lk)].concat(mids.map(function(v) { return cellToPixel(v.col, v.row); })).concat([[Number(lk.tx), Number(lk.ty)]]);
      const hit = findInsertSegment(poly, false, wx, wy);
      if (!hit) return false;
      const pt = worldPointToCellPoint(hit.near[0], hit.near[1], snapToGrid);
      pushUndo();
      if (!Array.isArray(lk.midVertices)) lk.midVertices = [];
      lk.midVertices.splice(Math.max(0, hit.insertIndex - 1), 0, pt);
      state.selectedVertex = { type: 'apronLink', id: lk.id, kind: 'mid', midIndex: Math.max(0, hit.insertIndex - 1) };
      if (typeof redrawLayoutAfterEdit === 'function') redrawLayoutAfterEdit();
      else if (typeof updateAllFlightPaths === 'function') updateAllFlightPaths(); else draw();
      return true;
    }
    return false;
  }

  function snapWorldPointToTaxiwayPolyline(wx, wy, taxiwayId) {
    const tw = (state.taxiways || []).find(t => t.id === taxiwayId);
    if (!tw || !tw.vertices || tw.vertices.length < 2) return null;
    const click = [wx, wy];
    let best = null;
    let bestD2 = Infinity;
    for (let i = 0; i < tw.vertices.length - 1; i++) {
      const [x1, y1] = cellToPixel(tw.vertices[i].col, tw.vertices[i].row);
      const [x2, y2] = cellToPixel(tw.vertices[i + 1].col, tw.vertices[i + 1].row);
      const near = closestPointOnSegment([x1, y1], [x2, y2], click);
      if (!near) continue;
      const d2 = dist2(near, click);
      if (d2 < bestD2) { bestD2 = d2; best = near; }
    }
    return best;
  }

  function hitTestApronLinkVertex(wx, wy) {
    if (!state.selectedObject || state.selectedObject.type !== 'apronLink') return null;
