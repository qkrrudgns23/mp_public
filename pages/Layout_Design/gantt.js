      allowedIcaoCategories: allowedIcaoCategories,
      allowedAircraftTypes: panelAllowedTypes,
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
    const uRm = readUnifiedNewStandConstraintFromPanel('remoteIcaoCategories', 'remoteAircraftAccess', ['A', 'B', 'C']);
    const categoryMode = uRm.categoryMode;
    const category = uRm.category;
    const allowedIcaoCategoriesR = uRm.allowedIcaoCategories;
    const panelAllowedTypesR = uRm.allowedAircraftTypes;
    const angleDeg = 0;
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
      allowedIcaoCategories: allowedIcaoCategoriesR,
      allowedAircraftTypes: panelAllowedTypesR,
      allowedTerminals: Array.from((document.getElementById('remoteTerminalAccess') || document).querySelectorAll('.remote-term-check')).filter(function(ch) { return ch.checked; }).map(function(ch) { return String(ch.getAttribute('data-item-id') || '').trim(); }).filter(Boolean)
    }));
    return true;
  }
  function tryPlaceTempStandAt(wx, wy) {
    const snap = snapTempStandOnTaxiwayCenterlines(wx, wy);
    if (!snap) return false;
    const sx = snap.x, sy = snap.y;
    const uTs = readUnifiedNewStandConstraintFromPanel('tempStandIcaoCategories', 'tempStandAircraftAccess', ['A', 'B', 'C']);
    const categoryMode = uTs.categoryMode;
    const category = uTs.category;
    const allowedIcaoCategoriesT = uTs.allowedIcaoCategories;
    const panelAllowedTypesT = uTs.allowedAircraftTypes;
    const angleDeg = 0;
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
      allowedIcaoCategories: allowedIcaoCategoriesT,
      allowedAircraftTypes: panelAllowedTypesT,
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
    if (tw.pathType === 'runway_exit' || tw.pathType === 'runway_taxiway') {
      copy.queueFlow = tw.queueFlow === true;
    } else {
      delete copy.queueFlow;
    }
    if (pathOpsEligiblePathType(tw.pathType)) {
      stripPathOpsDefaultsFromTaxiwaySerializeCopy(copy);
    } else {
      delete copy.pathOpsSlotOn;
      delete copy.pathOpsSlotCw;
      delete copy.pathOpsSlotCcw;
      delete copy.slotOn48;
      delete copy.slotCw48;
      delete copy.slotCcw48;
      delete copy.icaoCategoryAllowedMask;
    }
    return copy;
  }
  function partitionTaxiwaysForPersist(list) {
    const runwayPaths = [];
    const runwayTaxiways = [];
    const taxiways = [];
    (list || []).forEach(function(tw) {
      const ser = serializeTaxiwayWithEndpoints(tw);
      const pt = tw.pathType || 'taxiway';
      if (pt === 'runway') runwayPaths.push(ser);
      else if (pt === 'runway_exit' || pt === 'runway_taxiway') runwayTaxiways.push(ser);
      else {
        if (pt === 'general_queue_taxiway') ser.pathType = pt;
        else delete ser.pathType;
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
        layerMono: state.layerMono ? Object.assign({}, state.layerMono) : Object.assign({}, DEFAULT_LAYER_MONO),
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
          'timeMin',
          'sibtDate',
          'dwellMin',
          'minDwellMin',
          'noWayArr',
          'noWayDep',
          'arrRetFailed',
          'serviceDate',
          'sldtMin',
          'sibtMin',
          'sobtMin',
          'stotMin',
          'eldtMin',
          'eibtMin',
          'eobtMin',
          'etotMin',
          'arrApronId',
          'depApronId',
          'terminalId',
          'arrTerminalId',
          'depTerminalId',
          'eibtMinList',
          'eobtMinList',
          'ePushFinishedMinList',
          'arrRunwayDirUsed',
          'depRunwayDirUsed',
          'arrTdDistM',
          'arrVTdMs',
          'arrDecelMs2',
          'arrDep',
          'intDom',
          'arrRotSec',
          'proSimVttArrSec',
          'proSimDttArrSec',
          'proSimPushbackSec',
          'proSimDttDepSec',
          'proSimVttDepSec',
          'proSimDepLineupSec',
          'arrRunwayIdUsed',
          'arrRetDistM',
          'arrVRetInMs',
          'arrVRetOutMs',
        ];
        simFlightKeys.forEach(function(k) {
          if (Object.prototype.hasOwnProperty.call(f, k) && f[k] !== undefined) {
            copy[k] = f[k];
          }
        });
        const apronStaySegments = (typeof serializableApronStaySegmentsForFlight === 'function')
          ? serializableApronStaySegmentsForFlight(f)
          : [];
        if (apronStaySegments.length) {
          copy.apronStaySegments = apronStaySegments.map(function(seg) {
            const out = { sibtMin: seg.sibtMin, sobtMin: seg.sobtMin };
            if (seg.standId != null) out.standId = seg.standId;
            return out;
          });
          copy.arrApronId = copy.apronStaySegments[0].standId || null;
          copy.depApronId = copy.apronStaySegments[copy.apronStaySegments.length - 1].standId || null;
          copy.standId = copy.depApronId || null;
        }
        if (Array.isArray(f.edge_list) && f.edge_list.length) {
          copy.edge_list = f.edge_list.slice();
        }
        const t = f.token || {};
        const arrRwyId = f.arrRunwayId || t.arrRunwayId || t.runwayId || null;
        const apronId = (copy.depApronId != null ? copy.depApronId : (f.standId != null ? f.standId : (t.apronId != null ? t.apronId : null)));
        const termId = f.terminalId || t.terminalId || null;
        const arrTermId = f.arrTerminalId || t.arrTerminalId || termId || null;
        const depTermId = f.depTerminalId || t.depTerminalId || termId || null;
        const depRwyId = f.depRunwayId || t.depRunwayId || null;
        const exitTwId = (f.sampledArrRet != null && f.sampledArrRet !== '') ? f.sampledArrRet : (t.ExitTaxiwayId != null ? t.ExitTaxiwayId : null);
        copy.token = {
          arrRunwayId: arrRwyId,
          ExitTaxiwayId: exitTwId || null,
          apronId: apronId || null,
          terminalId: termId || null,
          arrTerminalId: arrTermId || null,
          depTerminalId: depTermId || null,
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
          arrTerminalId: _labelOrId(arrTermId, typeof getTerminalDisplayLabelById === 'function' ? getTerminalDisplayLabelById : null),
          depTerminalId: _labelOrId(depTermId, typeof getTerminalDisplayLabelById === 'function' ? getTerminalDisplayLabelById : null),
          depRunwayId: _labelOrId(depRwyId, typeof getRunwayDisplayLabelById === 'function' ? getRunwayDisplayLabelById : null),
        };
        const schedExport = flightScheduleMinutesForRow(f);
        copy.sibtDateTime = formatFlightScheduleDateTime(f, schedExport.sibt);
        copy.sobtDateTime = formatFlightScheduleDateTime(f, schedExport.sobt);
        copy.sldtDateTime = formatFlightScheduleDateTime(f, schedExport.sldt);
        copy.stotDateTime = formatFlightScheduleDateTime(f, schedExport.stot);
        if (state.hasSimulationResult && f.timeline_meta && typeof f.timeline_meta === 'object') {
          try {
            copy.timeline_meta = JSON.parse(JSON.stringify(f.timeline_meta));
          } catch (eMeta) {
            copy.timeline_meta = Object.assign({}, f.timeline_meta);
          }
        }
        if (state.hasSimulationResult && Array.isArray(f.proSimEdgeList) && f.proSimEdgeList.length) {
          copy.proSimEdgeList = f.proSimEdgeList.slice();
        }
        ensureFlightLookaheadArrDepFlight(f);
        copy.lookaheadArr = f.lookaheadArr;
        copy.lookaheadDep = f.lookaheadDep;
        return copy;
      }),
      layoutMarkers: (state.layoutMarkers || []).map(function(m) {
        if (!m || !m.kind) return null;
        if (m.kind === 'text') {
          return { kind: 'text', id: m.id, x: Number(m.x), y: Number(m.y), text: String(m.text || '') };
        }
        if (m.kind === 'ruler') {
          return { kind: 'ruler', id: m.id, x1: Number(m.x1), y1: Number(m.y1), x2: Number(m.x2), y2: Number(m.y2) };
        }
        if (m.kind === 'island') {
          const pts = Array.isArray(m.points) ? m.points.map(function(p) {
            return { x: Number(p && p.x), y: Number(p && p.y) };
          }).filter(function(p) { return isFinite(p.x) && isFinite(p.y); }) : [];
          if (pts.length < 3) return null;
          return {
            kind: 'island',
            id: m.id,
            points: pts,
            widthM: islandWidthMResolved(m),
          };
        }
        if (m.kind === 'area') {
          const pts = Array.isArray(m.points) ? m.points.map(function(p) {
            return { x: Number(p && p.x), y: Number(p && p.y) };
          }).filter(function(p) { return isFinite(p.x) && isFinite(p.y); }) : [];
          if (pts.length < 3) return null;
          return { kind: 'area', id: m.id, points: pts };
        }
        if (m.kind === 'flight') {
          const si = (typeof m.segIndex === 'number' && isFinite(m.segIndex)) ? Math.floor(m.segIndex) : (parseInt(m.segIndex, 10) || 0);
          return {
            kind: 'flight',
            id: m.id,
            taxiwayId: m.taxiwayId,
            segIndex: si,
            t: Number(m.t),
            aircraftType: String(m.aircraftType || '').trim(),
            blazerEnabled: !!m.blazerEnabled,
            headingReversed: !!m.headingReversed,
            blazerColor: MARKER_BLAZER_COLOR_OPTIONS.indexOf(String(m.blazerColor || '').trim()) >= 0 ? String(m.blazerColor).trim() : MARKER_BLAZER_COLOR_OPTIONS[0],
            blazerLeftTrail: Array.isArray(m.blazerLeftTrail) ? m.blazerLeftTrail.map(function(p) { return { x: Number(p && p.x), y: Number(p && p.y) }; }).filter(function(p) { return isFinite(p.x) && isFinite(p.y); }) : [],
            blazerRightTrail: Array.isArray(m.blazerRightTrail) ? m.blazerRightTrail.map(function(p) { return { x: Number(p && p.x), y: Number(p && p.y) }; }).filter(function(p) { return isFinite(p.x) && isFinite(p.y); }) : []
          };
        }
        if (m.kind === 'navaid') {
          return { kind: 'navaid', id: m.id, subType: (m.subType === 'ils') ? 'ils' : 'papi', x: Number(m.x), y: Number(m.y) };
        }
        return null;
      }).filter(Boolean),
