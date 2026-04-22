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
        if (m.kind === 'navaid') {
          return { kind: 'navaid', id: m.id, subType: (m.subType === 'ils') ? 'ils' : 'papi', x: Number(m.x), y: Number(m.y) };
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
    const tpl = typeof window.__GRID3D_VIEWER_HTML_TEMPLATE__ === 'string' ? window.__GRID3D_VIEWER_HTML_TEMPLATE__ : '';
    if (!tpl || tpl.length < 80) {
      console.error('Grid 3D viewer template missing');
      alert('3D viewer template is not loaded. Ensure pages/Layout_Design/3D/grid3d-viewer.html exists and reload the Layout Design page.');
      return;
    }
    const bootHtml = '<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/><title>Layout 3D</title>' +
      '<style>html,body{margin:0;height:100%;background:#0d0d0f;color:#e2e8f0;font-family:system-ui,sans-serif;overflow:hidden}' +
      '.wrap{display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:100vh;gap:18px;padding:24px;box-sizing:border-box}' +
      '.sp{width:44px;height:44px;border:3px solid rgba(148,163,184,.25);border-top-color:#7c6af7;border-radius:50%;animation:g .85s linear infinite}' +
      '@keyframes g{to{transform:rotate(360deg)}}' +
      '.bar{width:min(360px,86vw);height:4px;border-radius:2px;background:rgba(148,163,184,.2);overflow:hidden}' +
      '.bar>i{display:block;height:100%;width:38%;background:linear-gradient(90deg,#5b52d6,#7c6af7);border-radius:2px;animation:p 1.15s ease-in-out infinite}' +
      '@keyframes p{0%,100%{transform:translateX(-40%)}50%{transform:translateX(200%)}}' +
      '.t{font-size:15px;font-weight:600;color:#f1f5f9;text-align:center}.s{font-size:13px;color:#94a3b8;text-align:center;max-width:360px;line-height:1.45}' +
      '</style></head><body><div class="wrap"><div class="sp"></div><div class="bar"><i></i></div><p class="t">3D 뷰 준비 중</p>' +
      '<p class="s">레이아웃 스냅샷을 만들고 있습니다. 잠시만 기다려 주세요.</p></div></body></html>';
    const g3Base = (typeof GRID3D_ASSET_API_URL === 'string' && GRID3D_ASSET_API_URL.trim()) ? GRID3D_ASSET_API_URL.trim() : '';
    const viewerShellUrl = /^https?:\/\//i.test(g3Base) ? g3Base.replace(/\/$/, '') + '/api/grid3d-viewer-app' : '';
    let w = null;
    let openedViaReceiverShell = false;
    if (viewerShellUrl) {
      try {
        w = window.open(viewerShellUrl, '_blank', 'width=1280,height=840');
        openedViaReceiverShell = !!w;
      } catch (eHttp) {
        console.warn('Grid 3D receiver shell open failed', eHttp);
        w = null;
        openedViaReceiverShell = false;
      }
    }
    if (!w) {
      try {
        w = window.open('data:text/html;charset=utf-8,' + encodeURIComponent(bootHtml), '_blank', 'width=1280,height=840');
      } catch (eData) {
        console.warn('Grid 3D popup data URL failed, using about:blank', eData);
      }
    }
    if (!w) {
      w = window.open('about:blank', '_blank', 'width=1280,height=840');
    }
    if (!w) {
      alert('Popup was blocked. Allow popups for this site to open the 3D viewer.');
      return;
    }
    if (!openedViaReceiverShell) {
      var bootHref = '';
      try {
        bootHref = w.location && w.location.href ? String(w.location.href) : '';
      } catch (eLoc) {
        bootHref = '';
      }
      if (bootHref.indexOf('data:') !== 0) {
        try {
          w.document.open();
          w.document.write(bootHtml);
          w.document.close();
        } catch (eOpen) {
          console.error(eOpen);
          try {
            w.close();
          } catch (eClose) { /* ignore */ }
          alert('Could not open the 3D viewer window.');
          return;
        }
      }
    }
    let payload;
    try {
      payload = typeof buildLayout3DViewerPayload === 'function' ? buildLayout3DViewerPayload() : null;
    } catch (e) {
      console.error('buildLayout3DViewerPayload failed:', e);
      try {
        w.close();
      } catch (eClose2) { /* ignore */ }
      alert('Could not serialize layout for 3D: ' + (e && e.message ? e.message : e));
      return;
    }
    if (!payload || !payload.layout) {
      try {
        w.close();
      } catch (eClose3) { /* ignore */ }
      alert('Could not serialize layout for 3D.');
      return;
    }
    function sendGrid3dInit() {
      try {
        w.postMessage({ kind: 'grid3dViewerInit', payload: payload }, '*');
      } catch (e4) {
        console.error('postMessage to 3D viewer failed:', e4);
        alert('Could not send layout data to the 3D window. Try again or check the browser console.');
      }
    }
    if (!openedViaReceiverShell) {
      try {
        w.document.open();
        w.document.write(tpl);
        w.document.close();
      } catch (e3) {
        console.error(e3);
        try {
          w.close();
        } catch (eClose4) { /* ignore */ }
        alert('Could not write the 3D viewer document.');
        return;
      }
      setTimeout(sendGrid3dInit, 0);
    } else {
      function onShellReady() {
        setTimeout(sendGrid3dInit, 0);
      }
      try {
        if (w.document && w.document.readyState === 'complete') {
          onShellReady();
        } else {
          w.addEventListener('load', function grid3dShellLoad() {
            w.removeEventListener('load', grid3dShellLoad);
            onShellReady();
          });
        }
      } catch (eReady) {
        setTimeout(sendGrid3dInit, 150);
      }
    }
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
  function getPbbBoardingWidthM(pbb) {
    const w = Number(pbb && pbb.boardingWidthM);
    if (isFinite(w) && w > 0) return w;
    return 5;
  }
  function getPbbBoardingHeightM(pbb) {
    const h = Number(pbb && pbb.boardingHeightM);
    if (isFinite(h) && h > 0) return h;
    return 15;
  }
  function getPbbTerminalContactSetbackM(pbb) {
    const v = Number(pbb && pbb.terminalContactSetbackM);
    if (isFinite(v) && v >= 0) return v;
    return 0;
  }
  function getPbbTerminalFrameFromEdge(term, edgeIndex, wallX, wallY) {
    const v1 = term.vertices[edgeIndex], v2 = term.vertices[(edgeIndex + 1) % term.vertices.length];
    const p1 = cellToPixel(v1.col, v1.row), p2 = cellToPixel(v2.col, v2.row);
    const edx = p2[0] - p1[0], edy = p2[1] - p1[1];
    const el = Math.hypot(edx, edy) || 1;
    const tx = edx / el, ty = edy / el;
    let nx = -ty, ny = tx;
    let tcx = 0, tcy = 0;
    term.vertices.forEach(function(v) {
      const q = cellToPixel(v.col, v.row);
      tcx += q[0];
      tcy += q[1];
    });
    tcx /= term.vertices.length;
    tcy /= term.vertices.length;
    const inX = tcx - wallX, inY = tcy - wallY;
    if (nx * inX + ny * inY > 0) {
      nx = -nx;
      ny = -ny;
    }
    return { tx: tx, ty: ty, nx: nx, ny: ny };
  }
  function getPbbTerminalFrameAtWorld(wx, wy) {
    const proj = getClosestTerminalEdgePoint(wx, wy);
    if (!proj || !proj.term) return null;
    const fr = getPbbTerminalFrameFromEdge(proj.term, proj.edgeIndex, proj.point[0], proj.point[1]);
    return { mpx: proj.point[0], mpy: proj.point[1], term: proj.term, edgeIndex: proj.edgeIndex, tx: fr.tx, ty: fr.ty, nx: fr.nx, ny: fr.ny };
  }
  function ensurePbbBoardingWallGeometry(pbb) {
    if (!pbb || !Array.isArray(pbb.pbbBridges) || !pbb.pbbBridges.length) return;
    const Tsrcx = Number.isFinite(Number(pbb.x1)) ? Number(pbb.x1) : Number(pbb.pbbBridges[0].points[0].x) || 0;
    const Tsrcy = Number.isFinite(Number(pbb.y1)) ? Number(pbb.y1) : Number(pbb.pbbBridges[0].points[0].y) || 0;
    const proj = getClosestTerminalEdgePoint(Tsrcx, Tsrcy);
    if (!proj || !proj.term) return;
    const fr = getPbbTerminalFrameFromEdge(proj.term, proj.edgeIndex, proj.point[0], proj.point[1]);
    const wx = proj.point[0], wy = proj.point[1];
    const Tx = wx, Ty = wy;
    const depthM = getPbbBoardingHeightM(pbb);
    const Bx = Tx + fr.nx * depthM, By = Ty + fr.ny * depthM;
    pbb.pbbBridges.forEach(function(bridge) {
      if (!bridge.points || bridge.points.length < 3) return;
      bridge.points[0].x = Tx;
      bridge.points[0].y = Ty;
      bridge.points[1].x = Bx;
      bridge.points[1].y = By;
    });
    pbb.x1 = Tx;
    pbb.y1 = Ty;
    pbb.x2 = Bx;
    pbb.y2 = By;
  }
  function applyPbbArmLengthToBridgeEnds(pbb, armLenM) {
    if (!pbb || !Array.isArray(pbb.pbbBridges)) return;
    ensurePbbBoardingWallGeometry(pbb);
    const len = Math.max(3, Number(armLenM) || 15);
    pbb.pbbBridges.forEach(function(bridge) {
      const pts = bridge.points;
      if (!pts || pts.length < 3) return;
      const bx = Number(pts[1].x), by = Number(pts[1].y);
      const px = Number(pts[2].x), py = Number(pts[2].y);
      let vx = px - bx, vy = py - by;
      const hl = Math.hypot(vx, vy) || 1;
      vx /= hl;
      vy /= hl;
      pts[2].x = bx + vx * len;
      pts[2].y = by + vy * len;
    });
    bumpPathPolylineCacheRev();
  }
  function getPbbBoardingRectangleCornersWorldPx(pbb) {
    ensurePbbBoardingWallGeometry(pbb);
    const proj = getClosestTerminalEdgePoint(Number(pbb.x1) || 0, Number(pbb.y1) || 0);
    if (!proj || !proj.term) return null;
    const fr = getPbbTerminalFrameFromEdge(proj.term, proj.edgeIndex, proj.point[0], proj.point[1]);
    const wx = proj.point[0], wy = proj.point[1];
    const Tx = wx, Ty = wy;
    const halfW = getPbbBoardingWidthM(pbb) * 0.5;
    const depthM = getPbbBoardingHeightM(pbb);
    const c0 = [Tx - fr.tx * halfW, Ty - fr.ty * halfW];
    const c1 = [Tx + fr.tx * halfW, Ty + fr.ty * halfW];
    const c2 = [c1[0] + fr.nx * depthM, c1[1] + fr.ny * depthM];
    const c3 = [c0[0] + fr.nx * depthM, c0[1] + fr.ny * depthM];
    return [c0, c1, c2, c3];
  }
  function drawPbbBoardingRectangle(ctx, pbb, sel) {
    const poly = getPbbBoardingRectangleCornersWorldPx(pbb);
    if (!poly || poly.length < 4) return;
    const nowPerf = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
    const suppressStandFill = !!state.isPanning || nowPerf < _layoutDetailSuppressUntil;
    const sl = !!state.layers.standLines, sf = !!state.layers.standFill && !suppressStandFill;
    if (!sl && !sf) return;
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(poly[0][0], poly[0][1]);
    for (let i = 1; i < poly.length; i++) ctx.lineTo(poly[i][0], poly[i][1]);
    ctx.closePath();
    const hairW = layoutHairlineStrokeWidthWorld();
    if (sf) {
      ctx.fillStyle = sel ? 'rgba(255,255,255,0.2)' : 'rgba(255,255,255,0.14)';
      ctx.fill();
      drawPolygonDiagonalHatch45M(ctx, poly, LAYOUT_AREA_DIAGONAL_HATCH_SPACING_M, sel ? 'rgba(255,255,255,0.48)' : 'rgba(255,255,255,0.4)', hairW);
    }
    if (sl) {
      ctx.beginPath();
      ctx.moveTo(poly[0][0], poly[0][1]);
      for (let i = 1; i < poly.length; i++) ctx.lineTo(poly[i][0], poly[i][1]);
      ctx.closePath();
      ctx.strokeStyle = 'rgba(255,255,255,0.88)';
      ctx.lineWidth = hairW;
      ctx.stroke();
    }
    ctx.restore();
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
    return getStandApronTaxiwayAttachWorldPx(stand);
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
    const forward = [a].concat(mids).concat([b]);
    if (String(lk.apronDrawFirstEndpoint || 'stand') === 'taxiway' && forward.length >= 2) {
      return forward.slice().reverse();
    }
    return forward;
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
  function snapTempStandOnTaxiwayCenterlines(wx, wy) {
    const click = [wx, wy];
    const maxD2 = (CELL_SIZE * HIT_TW_SEG_CF) ** 2;
    let best = null;
    let bestD2 = maxD2;
    const allowPt = { taxiway: 1, runway_exit: 1, runway_taxiway: 1, general_queue_taxiway: 1 };
    (state.taxiways || []).forEach(function(tw) {
      const pt = tw.pathType || 'taxiway';
      if (!allowPt[pt]) return;
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
    const hitPadSq = Math.pow(Math.max(5, CELL_SIZE * 0.22), 2);
    const lineW = c2dHoldingPointMarkingLineWidthWorld();
    const pairHalf = holdingPointMarkingDoubleLineGapM(lineW) * 0.5;
    const pts = state.holdingPoints || [];
    for (let i = pts.length - 1; i >= 0; i--) {
      const hp = pts[i];
      if (!hp || !isFinite(hp.x) || !isFinite(hp.y)) continue;
      const g = findHoldingPointPathGeometry(hp);
      const perp = holdingPointPerpFromTangent(g.ux, g.uy);
      const halfLen = holdingPointBarHalfLengthMFromPathWidth(g.pathWidthM);
      const cx = hp.x, cy = hp.y;
      const nearOfs = function(ofs) {
        const x0 = cx - perp.px * halfLen + g.ux * ofs, y0 = cy - perp.py * halfLen + g.uy * ofs;
        const x1 = cx + perp.px * halfLen + g.ux * ofs, y1 = cy + perp.py * halfLen + g.uy * ofs;
        return distPointToSegmentSq(wx, wy, x0, y0, x1, y1) <= hitPadSq;
      };
      if (nearOfs(-pairHalf) || nearOfs(pairHalf)) return { type: 'holdingPoint', id: hp.id, obj: hp };
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
      if (isFinite(x) && isFinite(y) && dist2(click, [x, y]) <= r2)
        return { markerId: mk.id, handle: 'navaidCenter' };
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
