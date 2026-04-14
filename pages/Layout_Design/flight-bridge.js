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
        if (pt === 'apron_taxiway') ser.pathType = 'apron_taxiway';
        else if (pt === 'general_queue_taxiway') ser.pathType = 'general_queue_taxiway';
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
        const gj = buildPathGraph(null);
        const cj = (gj && (gj.connectedJunctions || gj.junctions)) || [];
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
          'timeMin',
          'sibtDate',
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
        const schedExport = flightScheduleMinutesForRow(f);
        copy.sibtDateTime = formatFlightScheduleDateTime(f, schedExport.sibt);
        copy.sobtDateTime = formatFlightScheduleDateTime(f, schedExport.sobt);
        copy.sldtDateTime_d = formatFlightScheduleDateTime(f, schedExport.sldt_d);
        copy.sibtDateTime_d = formatFlightScheduleDateTime(f, schedExport.sibt_d);
        copy.sobtDateTime_d = formatFlightScheduleDateTime(f, schedExport.sobt_d);
        copy.stotDateTime_d = formatFlightScheduleDateTime(f, schedExport.stot_d);
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
            outerWidthM: islandOuterWidthMResolved(m),
            innerWidthM: islandInnerWidthMResolved(m),
            pavement: islandMarkerPavementResolved(m)
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
        return null;
      }).filter(Boolean),
      simPathGraph: buildSimPathGraphExport()
    };
  }
  function buildLayout3DViewerPayload() {
    let tSec = Number(state.simTimeSec);
    if (!isFinite(tSec)) tSec = 0;
    const layout = serializeCurrentLayout();
    const flightDrawPoses = [];
    (state.flights || []).forEach(function(f) {
      if (!f) return;
      let pose = null;
      if (typeof getFlightPoseAtTimeForDraw === 'function') {
        pose = getFlightPoseAtTimeForDraw(f, tSec);
      }
      flightDrawPoses.push({
        id: f.id,
        reg: f.reg,
        aircraftType: f.aircraftType,
        code: f.code,
        arrDep: f.arrDep,
        pose: pose && isFinite(pose.x) && isFinite(pose.y) ? { x: pose.x, y: pose.y, dx: pose.dx, dy: pose.dy } : null
      });
    });
    const enrichedFootprints = {
      remote: (state.remoteStands || []).map(function(st) {
        return {
          id: st && st.id,
          name: st && st.name,
          corners: typeof getRemoteStandCorners === 'function' ? getRemoteStandCorners(st) : null
        };
      }).filter(function(r) { return r.corners && r.corners.length >= 3; }),
      pbb: (state.pbbStands || []).map(function(pbb) {
        return {
          id: pbb && pbb.id,
          name: pbb && pbb.name,
          corners: typeof getPBBStandCorners === 'function' ? getPBBStandCorners(pbb) : null
        };
      }).filter(function(r) { return r.corners && r.corners.length >= 3; })
    };
    const enrichedApronLinkPolylines = (state.apronLinks || []).map(function(lk) {
      if (!lk || typeof getApronLinkPolylineWorldPts !== 'function') return null;
      const pts = getApronLinkPolylineWorldPts(lk);
      if (!pts || pts.length < 2) return null;
      return {
        id: lk.id,
        points: pts.map(function(p) { return { x: p[0], y: p[1] }; })
      };
    }).filter(Boolean);
    const payload = {
      version: 1,
      kind: 'grid3dViewer',
      layoutApiUrl: (typeof LAYOUT_API_URL === 'string' && LAYOUT_API_URL) ? LAYOUT_API_URL : '',
      grid3dAssetApiUrl: (typeof GRID3D_ASSET_API_URL === 'string' && GRID3D_ASSET_API_URL) ? GRID3D_ASSET_API_URL : '',
      exportedAt: new Date().toISOString(),
      simTimeSec: tSec,
      viewerConfig: {
        gridMajorInterval: GRID_MAJOR_INTERVAL,
        gridViewBg: GRID_VIEW_BG
      },
      layout: layout,
      flightDrawPoses: flightDrawPoses,
      enrichedFootprints: enrichedFootprints,
      enrichedApronLinkPolylines: enrichedApronLinkPolylines
    };
    try {
      let tiled = null;
      if (typeof exportLayoutGroundTilesFor3D === 'function') tiled = exportLayoutGroundTilesFor3D();
      if (tiled && tiled.tiles && tiled.tiles.length === 4) {
        payload.layoutGroundTiles = tiled;
      } else if (typeof exportLayoutGroundTextureFor3D === 'function') {
        const gt = exportLayoutGroundTextureFor3D();
        if (gt && gt.dataUrl) payload.layoutGroundTexture = gt;
      }
    } catch (eTex) {
      console.warn('exportLayoutGroundTilesFor3D / exportLayoutGroundTextureFor3D failed', eTex);
    }
    return payload;
  }
  function openGrid3DViewerWindow() {
