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

  function hitTestSimFlightAtWorld(wx, wy) {
    if (!state.hasSimulationResult || !state.flights || !state.flights.length) return null;
    if (!state.globalUpdateFresh) return null;
    const tSec = state.simTimeSec;
    if (typeof prepareLazyTimelinesForCurrentSim === 'function') prepareLazyTimelinesForCurrentSim(tSec);
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
    const lk = state.selectedObject.obj;
    if (!lk || lk.id !== state.selectedObject.id) return null;
    const click = [wx, wy];
    const maxD2 = (CELL_SIZE * HIT_TW_VTX_CF) ** 2;
    let best = null;
    let bestD2 = maxD2;
    const tx = Number(lk.tx), ty = Number(lk.ty);
    if (isFinite(tx) && isFinite(ty)) {
      const d2 = dist2([tx, ty], click);
      if (d2 < bestD2) { bestD2 = d2; best = { linkId: lk.id, kind: 'taxiway' }; }
    }
    (lk.midVertices || []).forEach((v, idx) => {
      const [vx, vy] = cellToPixel(Number(v.col), Number(v.row));
      const d2 = dist2([vx, vy], click);
      if (d2 < bestD2) { bestD2 = d2; best = { linkId: lk.id, kind: 'mid', midIndex: idx }; }
    });
    return best;
  }

  function isSelectedVertex(type, objectId, index) {
    const sv = state.selectedVertex;
    return !!(sv && sv.type === type && sv.id === objectId && sv.index === index);
  }

  function removeSelectedVertex() {
    const sv = state.selectedVertex;
    if (!sv) return false;
    if (sv.type === 'terminal') {
      const term = state.terminals.find(t => t.id === sv.id);
      if (!term || !Array.isArray(term.vertices) || sv.index < 0 || sv.index >= term.vertices.length) return false;
      if (term.closed && term.vertices.length <= 3) return false;
      pushUndo();
      term.vertices.splice(sv.index, 1);
      if (term.vertices.length < 3) term.closed = false;
      state.selectedVertex = null;
      if (state.currentTerminalId === term.id) syncPanelFromState();
      updateObjectInfo();
      draw();
      return true;
    }
    if (sv.type === 'taxiway') {
      const tw = state.taxiways.find(t => t.id === sv.id);
      if (!tw || !Array.isArray(tw.vertices) || sv.index < 0 || sv.index >= tw.vertices.length) return false;
      if (tw.vertices.length <= 2) return false;
      pushUndo();
      tw.vertices.splice(sv.index, 1);
      if (typeof syncStartEndFromVertices === 'function' && tw.vertices.length >= 2) syncStartEndFromVertices(tw);
      state.selectedVertex = null;
      syncPanelFromState();
      updateObjectInfo();
      if (typeof redrawLayoutAfterEdit === 'function') redrawLayoutAfterEdit();
      else if (typeof updateAllFlightPaths === 'function') updateAllFlightPaths(); else draw();
      return true;
    }
    if (sv.type === 'apronLink') {
      if (sv.kind !== 'mid') return false;
      const lk = state.apronLinks.find(l => l.id === sv.id);
      if (!lk || !Array.isArray(lk.midVertices) || sv.midIndex < 0 || sv.midIndex >= lk.midVertices.length) return false;
      pushUndo();
      lk.midVertices.splice(sv.midIndex, 1);
      if (!lk.midVertices.length) delete lk.midVertices;
      state.selectedVertex = null;
      updateObjectInfo();
      if (typeof redrawLayoutAfterEdit === 'function') redrawLayoutAfterEdit();
      else if (typeof updateAllFlightPaths === 'function') updateAllFlightPaths(); else draw();
      return true;
    }
    return false;
  }

  function removeLastDrawingVertex() {
    if (state.terminalDrawingId) {
      const term = state.terminals.find(t => t.id === state.terminalDrawingId);
      if (!term || !Array.isArray(term.vertices) || !term.vertices.length) return false;
      pushUndo();
      term.vertices.pop();
      if (!term.vertices.length) state.layoutPathDrawPointer = null;
      state.selectedVertex = null;
      syncPanelFromState();
      updateObjectInfo();
      draw();
      return true;
    }
    if (state.taxiwayDrawingId) {
      const tw = state.taxiways.find(t => t.id === state.taxiwayDrawingId);
      if (!tw || !Array.isArray(tw.vertices) || !tw.vertices.length) return false;
      pushUndo();
      tw.vertices.pop();
      if (!tw.vertices.length) state.layoutPathDrawPointer = null;
      if (typeof syncStartEndFromVertices === 'function' && tw.vertices.length >= 2) syncStartEndFromVertices(tw);
      else {
        tw.start_point = null;
        tw.end_point = null;
      }
      state.selectedVertex = null;
      syncPanelFromState();
      updateObjectInfo();
      if (typeof redrawLayoutAfterEdit === 'function') redrawLayoutAfterEdit();
      else if (typeof updateAllFlightPaths === 'function') updateAllFlightPaths(); else draw();
      return true;
    }
    if (settingModeSelect.value === 'apronTaxiway' && state.apronLinkDrawing && state.apronLinkTemp) {
      if (state.apronLinkMidpoints && state.apronLinkMidpoints.length) {
        state.apronLinkMidpoints.pop();
        draw();
        return true;
      }
      state.apronLinkTemp = null;
      state.apronLinkMidpoints = [];
      state.apronLinkPointerWorld = null;
      draw();
      return true;
    }
    return false;
  }

  function getCurrentTerminal() {
    if (state.selectedObject && state.selectedObject.type === 'terminal' && state.selectedObject.obj) {


      return state.selectedObject.obj;
    }
    if (state.currentTerminalId) {
      const t = state.terminals.find(x => x.id === state.currentTerminalId);
      if (t) return t;
    }
    return state.terminals[0] || null;
  }

  function polygonAreaM2(vertices) {
    if (!vertices || vertices.length < 3) return 0;
    let area = 0;
    const n = vertices.length;
    for (let i = 0; i < n; i++) {
      const j = (i + 1) % n;
      area += vertices[i].col * vertices[j].row;
      area -= vertices[j].col * vertices[i].row;
    }
    return Math.abs(area) * 0.5 * CELL_SIZE * CELL_SIZE;
  }

  function syncPanelFromState() {
    document.getElementById('gridCellSize').value = CELL_SIZE;
    document.getElementById('gridCols').value = GRID_COLS;
    document.getElementById('gridRows').value = GRID_ROWS;
    const gridImageOpacityEl = document.getElementById('gridLayoutImageOpacity');
    const gridImageWidthEl = document.getElementById('gridLayoutImageWidthM');
    const gridImageHeightEl = document.getElementById('gridLayoutImageHeightM');
    const gridImageColEl = document.getElementById('gridLayoutImageCol');
    const gridImageRowEl = document.getElementById('gridLayoutImageRow');
    const gridImageMetaEl = document.getElementById('gridLayoutImageMeta');
    const gridImageClearBtn = document.getElementById('btnClearGridLayoutImage');
    const gridImageFileEl = document.getElementById('gridLayoutImageFile');
    const overlay = state.layoutImageOverlay;
    if (gridImageOpacityEl) gridImageOpacityEl.value = overlay ? String(overlay.opacity) : String(GRID_LAYOUT_IMAGE_DEFAULTS.opacity);
    if (gridImageWidthEl) gridImageWidthEl.value = overlay ? String(overlay.widthM) : String(GRID_LAYOUT_IMAGE_DEFAULTS.widthM);
    if (gridImageHeightEl) gridImageHeightEl.value = overlay ? String(overlay.heightM) : String(GRID_LAYOUT_IMAGE_DEFAULTS.heightM);
    if (gridImageColEl) gridImageColEl.value = overlay ? String(overlay.topLeftCol) : String(GRID_LAYOUT_IMAGE_DEFAULTS.topLeftCol);
    if (gridImageRowEl) gridImageRowEl.value = overlay ? String(overlay.topLeftRow) : String(GRID_LAYOUT_IMAGE_DEFAULTS.topLeftRow);
    if (gridImageMetaEl) gridImageMetaEl.textContent = overlay ? ('Loaded: ' + (overlay.name || 'Layout image')) : 'No file selected.';
    if (gridImageClearBtn) gridImageClearBtn.disabled = !overlay;
    if (!overlay && gridImageFileEl) gridImageFileEl.value = '';
    if (state.terminals.length && (!state.currentTerminalId || !state.terminals.some(t => t.id === state.currentTerminalId)))
      state.currentTerminalId = state.terminals[0].id;
    const term = getCurrentTerminal();
    if (term) {
      const buildingTypeSel = document.getElementById('buildingType');
      if (buildingTypeSel) {
        buildingTypeSel.innerHTML = getBuildingTypeOptionsHtml(term.buildingType);
        buildingTypeSel.value = normalizeBuildingType(term.buildingType);
      }
      document.getElementById('terminalName').value = term.name || getDefaultBuildingNameForType(term.buildingType, term.id);
      const floors = term.floors != null ? Math.max(1, parseInt(term.floors, 10) || 1) : 1;
      const f2fRaw = term.floorToFloor != null ? Number(term.floorToFloor) : (term.floorHeight != null ? Number(term.floorHeight) : 4);
      const f2f = Math.max(0.5, f2fRaw || 4);
      const totalH = term.floorHeight != null ? Number(term.floorHeight) || (floors * f2f) : (floors * f2f);
      term.floors = floors;
      term.floorToFloor = f2f;
      term.floorHeight = totalH;
      const floorsInput = document.getElementById('terminalFloors');
      const f2fInput = document.getElementById('terminalFloorToFloor');
      const totalInput = document.getElementById('terminalFloorHeight');
      if (floorsInput) floorsInput.value = floors;
      if (f2fInput) f2fInput.value = f2f;
      if (totalInput) totalInput.value = totalH;
      document.getElementById('terminalDepartureCapacity').value = term.departureCapacity != null ? term.departureCapacity : 0;
      document.getElementById('terminalArrivalCapacity').value = term.arrivalCapacity != null ? term.arrivalCapacity : 0;
    }
    syncDrawToggleButton('btnTerminalDraw', !!state.terminalDrawingId);
    if (state.selectedObject && state.selectedObject.type === 'pbb') {
      const pbb = state.selectedObject.obj;
      const nameInput = document.getElementById('standName');
      const modeSel = document.getElementById('standCategoryMode');
      const catSel = document.getElementById('standCategory');
      const lenInput = document.getElementById('pbbLength');
      const angleInput = document.getElementById('standAngle');
      const pbbCountInput = document.getElementById('pbbBridgeCount');
      const boardingWInput = document.getElementById('pbbBoardingWidth');
      const boardingHInput = document.getElementById('pbbBoardingHeight');
      if (nameInput) nameInput.value = pbb.name || '';
      if (modeSel) modeSel.value = getStandCategoryMode(pbb);
      if (catSel) catSel.value = pbb.category || 'C';
      if (lenInput) {
        let arm = Number(pbb.pbbArmLenM);
        if (!isFinite(arm) || arm <= 0) {
          const br0 = pbb.pbbBridges && pbb.pbbBridges[0];
          const p1 = br0 && br0.points && br0.points[1], p2 = br0 && br0.points && br0.points[2];
          if (p1 && p2) arm = Math.hypot(Number(p2.x) - Number(p1.x), Number(p2.y) - Number(p1.y));
          else arm = 15;
        }
        lenInput.value = String(Math.max(1, Math.round(arm)));
      }
      if (angleInput) angleInput.value = String(Math.round(getPbbAngleDeg(pbb)));
      if (pbbCountInput) pbbCountInput.value = String(Math.max(1, parseInt(pbb.pbbCount, 10) || 1));
      if (boardingWInput) boardingWInput.value = String(typeof getPbbBoardingWidthM === 'function' ? getPbbBoardingWidthM(pbb) : (Number(pbb.boardingWidthM) > 0 ? Number(pbb.boardingWidthM) : 5));
      if (boardingHInput) boardingHInput.value = String(typeof getPbbBoardingHeightM === 'function' ? getPbbBoardingHeightM(pbb) : (Number(pbb.boardingHeightM) > 0 ? Number(pbb.boardingHeightM) : 15));
      syncStandConstraintVisibility('stand', getStandCategoryMode(pbb));
      renderAircraftConstraintChoices('standAircraftAccess', getStandAllowedAircraftTypes(pbb));
    }
    if (state.selectedObject && state.selectedObject.type === 'remote') {
      const st = state.selectedObject.obj;
      const nameInput = document.getElementById('remoteName');
      const angleInput = document.getElementById('remoteAngle');
      const modeSel = document.getElementById('remoteCategoryMode');
      const catSel = document.getElementById('remoteCategory');
      if (nameInput) nameInput.value = st.name || '';
      if (angleInput) angleInput.value = String(Math.round(normalizeAngleDeg(st.angleDeg != null ? st.angleDeg : 0)));
      if (modeSel) modeSel.value = getStandCategoryMode(st);
      if (catSel) catSel.value = st.category || 'C';
      syncStandConstraintVisibility('remote', getStandCategoryMode(st));
      renderAircraftConstraintChoices('remoteAircraftAccess', getStandAllowedAircraftTypes(st));
      renderRemoteTerminalAccessChoices(Array.isArray(st.allowedTerminals) ? st.allowedTerminals : []);
    }
    if (state.selectedObject && state.selectedObject.type === 'holdingPoint') {
      const hp = state.selectedObject.obj;
      const nameInput = document.getElementById('holdingPointName');
      if (nameInput) nameInput.value = hp.name || '';
    }
    if (state.selectedObject && state.selectedObject.type === 'taxiway') {
      const tw = state.selectedObject.obj;
      const nameInput = document.getElementById('taxiwayName');
      const widthInput = document.getElementById('taxiwayWidth');
      const maxExitInput = document.getElementById('taxiwayMaxExitVel');
      const minExitInput = document.getElementById('taxiwayMinExitVel');
      if (nameInput) nameInput.value = tw.name || '';
      const widthDefault = tw.pathType === 'runway'
        ? RUNWAY_PATH_DEFAULT_WIDTH
        : (tw.pathType === 'runway_exit' ? RUNWAY_EXIT_DEFAULT_WIDTH : TAXIWAY_DEFAULT_WIDTH);
      if (widthInput) widthInput.value = tw.width != null ? tw.width : widthDefault;
      const avgVelInput = document.getElementById('taxiwayAvgMoveVelocity');
      if (avgVelInput) avgVelInput.value = (tw.avgMoveVelocity != null ? tw.avgMoveVelocity : 10);
      syncPathFieldVisibilityForPathType(tw.pathType || 'taxiway');
      const runwayMinArrInput = document.getElementById('runwayMinArrVelocity');
      if (runwayMinArrInput) {
        const mav = (typeof tw.minArrVelocity === 'number' && isFinite(tw.minArrVelocity) && tw.minArrVelocity > 0)
          ? Math.max(1, Math.min(150, tw.minArrVelocity))
          : 15;
        runwayMinArrInput.value = mav;
      }
      const runwayLineupInput = document.getElementById('runwayLineupDistM');
      if (runwayLineupInput && tw.pathType === 'runway') {
        const lv = getEffectiveRunwayLineupDistM(tw);
        runwayLineupInput.value = String(lv);
      }
      const runwayStartDispInput = document.getElementById('runwayStartDisplacedThresholdM');
      if (runwayStartDispInput && tw.pathType === 'runway') runwayStartDispInput.value = String(getEffectiveRunwayStartDisplacedThresholdM(tw));
      const runwayStartBlastInput = document.getElementById('runwayStartBlastPadM');
      if (runwayStartBlastInput && tw.pathType === 'runway') runwayStartBlastInput.value = String(getEffectiveRunwayStartBlastPadM(tw));
      const runwayEndDispInput = document.getElementById('runwayEndDisplacedThresholdM');
      if (runwayEndDispInput && tw.pathType === 'runway') runwayEndDispInput.value = String(getEffectiveRunwayEndDisplacedThresholdM(tw));
      const runwayEndBlastInput = document.getElementById('runwayEndBlastPadM');
      if (runwayEndBlastInput && tw.pathType === 'runway') runwayEndBlastInput.value = String(getEffectiveRunwayEndBlastPadM(tw));
      if (maxExitInput) maxExitInput.value = tw.maxExitVelocity != null ? tw.maxExitVelocity : 30;
      if (minExitInput) {
        const minVal = (typeof tw.minExitVelocity === 'number' && isFinite(tw.minExitVelocity) && tw.minExitVelocity > 0)
          ? tw.minExitVelocity
          : 15;
        minExitInput.value = minVal;
      }
      if (tw.pathType === 'runway_exit') {
        const allow = getTaxiwayAllowedRunwayDirections(tw);
        renderRunwayDirectionChoices(allow);
      } else {
        renderRunwayDirectionChoices([]);
      }
      const modeSel = document.getElementById('taxiwayDirectionMode');
      let d = getTaxiwayDirection(tw);
      if (tw.pathType === 'runway' && d === 'both') d = 'clockwise';
      if (modeSel) modeSel.value = d;
    } else if (state.selectedObject && state.selectedObject.type === 'apronLink') {
      const lk = state.selectedObject.obj;
      const nameInput = document.getElementById('apronLinkName');
      if (nameInput) nameInput.value = getApronLinkDisplayName(lk);
    } else if (state.selectedObject && state.selectedObject.type === 'layoutEdge') {
      const ed = state.selectedObject.obj;
      const nameInput = document.getElementById('edgeName');
      if (nameInput) nameInput.value = getLayoutEdgeDisplayName(ed);
    } else {
      const rm = settingModeSelect ? settingModeSelect.value : '';
      if (isPathLayoutMode(rm)) {
        const ptx = pathTypeFromLayoutMode(rm);
        syncPathFieldVisibilityForPathType(ptx);
        if (ptx === 'runway_exit') {
          const allowDef = (RW_EXIT_ALLOWED_DEFAULT && RW_EXIT_ALLOWED_DEFAULT.length) ? RW_EXIT_ALLOWED_DEFAULT : ['clockwise', 'counter_clockwise'];
          renderRunwayDirectionChoices(allowDef);
        }
      }
      else {
        const maxExitWrap = document.getElementById('runwayMaxExitVelWrap');
        if (maxExitWrap) maxExitWrap.style.display = 'none';
        const minExitWrap = document.getElementById('runwayMinExitVelWrap');
        if (minExitWrap) minExitWrap.style.display = 'none';
        const runwayMinArrWrap = document.getElementById('runwayMinArrVelocityWrap');
        if (runwayMinArrWrap) runwayMinArrWrap.style.display = 'none';
        const runwayLineupWrap = document.getElementById('runwayLineupDistWrap');
        if (runwayLineupWrap) runwayLineupWrap.style.display = 'none';
        const runwayStartDispWrap = document.getElementById('runwayStartDisplacedThresholdWrap');
        if (runwayStartDispWrap) runwayStartDispWrap.style.display = 'none';
        const runwayStartBlastWrap = document.getElementById('runwayStartBlastPadWrap');
        if (runwayStartBlastWrap) runwayStartBlastWrap.style.display = 'none';
        const runwayEndDispWrap = document.getElementById('runwayEndDisplacedThresholdWrap');
        if (runwayEndDispWrap) runwayEndDispWrap.style.display = 'none';
        const runwayEndBlastWrap = document.getElementById('runwayEndBlastPadWrap');
        if (runwayEndBlastWrap) runwayEndBlastWrap.style.display = 'none';
        const taxiwayAvgWrap = document.getElementById('taxiwayAvgVelocityWrap');
        if (taxiwayAvgWrap) taxiwayAvgWrap.style.display = 'none';
        const rwDirWrap = document.getElementById('runwayExitAllowedDirectionWrap');
        if (rwDirWrap) rwDirWrap.style.display = 'none';
      }
