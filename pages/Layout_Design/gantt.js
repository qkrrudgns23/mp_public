      for (let i = 0; i < segCount; i++) {
        const j = t.closed ? ((i + 1) % n) : (i + 1);
        considerSeg(cellToPixel(verts[i].col, verts[i].row), cellToPixel(verts[j].col, verts[j].row));
      }
    });
    (state.taxiways || []).forEach(function(tw) {
      if (!tw) return;
      const poly = typeof getOrderedPoints === 'function' ? getOrderedPoints(tw) : getTaxiwayOrderedPoints(tw);
      if (!poly || poly.length < 2) return;
      for (let i = 0; i < poly.length - 1; i++) considerSeg(poly[i], poly[i + 1]);
    });
    (state.holdingPoints || []).forEach(function(hp) {
      if (!hp || !isFinite(hp.x) || !isFinite(hp.y)) return;
      considerPoint([hp.x, hp.y]);
    });
    (state.apronLinks || []).forEach(function(lk) {
      const poly = typeof getApronLinkPolylineWorldPts === 'function' ? getApronLinkPolylineWorldPts(lk) : null;
      if (!poly || poly.length < 2) return;
      for (let i = 0; i < poly.length - 1; i++) considerSeg(poly[i], poly[i + 1]);
    });
    (state.pbbStands || []).forEach(function(pbb) {
      const corners = typeof getPBBStandCorners === 'function' ? getPBBStandCorners(pbb) : null;
      if (!corners || corners.length < 2) return;
      const m = corners.length;
      for (let i = 0; i < m; i++) considerSeg(corners[i], corners[(i + 1) % m]);
    });
    (state.remoteStands || []).forEach(function(st) {
      const corners = typeof getRemoteStandCorners === 'function' ? getRemoteStandCorners(st) : null;
      if (!corners || corners.length < 2) return;
      const m = corners.length;
      for (let i = 0; i < m; i++) considerSeg(corners[i], corners[(i + 1) % m]);
    });
    (state.tempStands || []).forEach(function(st) {
      const corners = typeof getRemoteStandCorners === 'function' ? getRemoteStandCorners(st) : null;
      if (!corners || corners.length < 2) return;
      const m = corners.length;
      for (let i = 0; i < m; i++) considerSeg(corners[i], corners[(i + 1) % m]);
    });
    return best == null ? null : { pt: best, d2: bestD2 };
  }
  function markerAreaSnapWorldToPlacementPx(wx, wy, snapToGrid) {
    const click = [wx, wy];
    const gridPx = worldPointToPixel(wx, wy, snapToGrid);
    const maxD2 = Math.pow(CELL_SIZE * HIT_TW_SEG_CF, 2);
    const pack = snapWorldPointToLayoutObjectsForMarker(wx, wy);
    if (pack && pack.d2 <= maxD2 && pack.d2 <= dist2(gridPx, click)) return pack.pt;
    return gridPx;
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
      const px = v && isFinite(Number(v.x)) && isFinite(Number(v.y))
        ? [Number(v.x), Number(v.y)]
        : cellToPixel(Number(v.col), Number(v.row));
      const d2 = dist2(px, click);
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
      markApronLinkJunctionOverlayDirty(lk.id);
      if (typeof redrawLayoutAfterEdit === 'function') redrawLayoutAfterEdit();
      else if (typeof updateAllFlightPaths === 'function') updateAllFlightPaths(); else draw();
      return true;
    }
    if (sv.type === 'layoutMarkerHandle' && sv.handle === 'islandVertex') {
      const mk = (state.layoutMarkers || []).find(function(m) { return m && String(m.id) === String(sv.id); });
      if (!mk || !isLayoutPolygonMarkerKind(mk.kind) || !Array.isArray(mk.points)) return false;
      const idx = sv.vertexIndex;
      if (typeof idx !== 'number' || idx < 0 || idx >= mk.points.length || mk.points.length <= 3) return false;
      pushUndo();
      mk.points.splice(idx, 1);
      state.selectedVertex = null;
      if (typeof updateObjectInfo === 'function') updateObjectInfo();
      draw();
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
    if (settingModeSelect.value === 'marker' && state.markerDrawing && getMarkerSubKindFromPanel() === 'island' && state.markerIslandDraft && state.markerIslandDraft.points && state.markerIslandDraft.points.length) {
      state.markerIslandDraft.points.pop();
      if (!state.markerIslandDraft.points.length) state.markerIslandDraft = null;
      state.markerIslandHoverWorld = null;
      draw();
      return true;
    }
    if (settingModeSelect.value === 'marker' && state.markerDrawing && getMarkerSubKindFromPanel() === 'area' && state.markerAreaDraft && state.markerAreaDraft.points && state.markerAreaDraft.points.length) {
      state.markerAreaDraft.points.pop();
      if (!state.markerAreaDraft.points.length) state.markerAreaDraft = null;
      state.markerAreaHoverWorld = null;
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
    return null;
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
    if (gridImageMetaEl) gridImageMetaEl.textContent = overlay ? ('Loaded : ' + (overlay.name || 'Layout image')) : 'No file selected';
    if (gridImageClearBtn) gridImageClearBtn.disabled = !overlay;
    if (!overlay && gridImageFileEl) gridImageFileEl.value = '';
    if (state.currentTerminalId && !state.terminals.some(t => t.id === state.currentTerminalId))
      state.currentTerminalId = null;
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
      const lenInput = document.getElementById('pbbLength');
      const angleInput = document.getElementById('standAngle');
      const pbbCountInput = document.getElementById('pbbBridgeCount');
      if (nameInput) nameInput.value = pbb.name || '';
      applyIcaoCategoriesToHost('standIcaoCategories', normalizeAllowedIcaoCategories(pbb.allowedIcaoCategories));
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
      const boardingWInput = document.getElementById('pbbBoardingWidth');
      const boardingHInput = document.getElementById('pbbBoardingHeight');
      if (boardingWInput) boardingWInput.value = String(getPbbBoardingWidthM(pbb));
      if (boardingHInput) boardingHInput.value = String(getPbbBoardingHeightM(pbb));
      syncStandConstraintVisibility('stand');
      renderAircraftConstraintChoices('standAircraftAccess', getStandAllowedAircraftTypes(pbb), pbb.allowedIcaoCategories);
    }
    if (state.selectedObject && state.selectedObject.type === 'remote') {
      const st = state.selectedObject.obj;
      const nameInput = document.getElementById('remoteName');
      if (nameInput) nameInput.value = st.name || '';
      applyIcaoCategoriesToHost('remoteIcaoCategories', normalizeAllowedIcaoCategories(st.allowedIcaoCategories));
      syncStandConstraintVisibility('remote');
      renderAircraftConstraintChoices('remoteAircraftAccess', getStandAllowedAircraftTypes(st), st.allowedIcaoCategories);
      renderRemoteTerminalAccessChoices(Array.isArray(st.allowedTerminals) ? st.allowedTerminals : []);
    }
    if (state.selectedObject && state.selectedObject.type === 'tempStand') {
      const st = state.selectedObject.obj;
      const nameInput = document.getElementById('tempStandName');
      if (nameInput) nameInput.value = st.name || '';
      applyIcaoCategoriesToHost('tempStandIcaoCategories', normalizeAllowedIcaoCategories(st.allowedIcaoCategories));
      syncStandConstraintVisibility('tempStand');
      renderAircraftConstraintChoices('tempStandAircraftAccess', getStandAllowedAircraftTypes(st), st.allowedIcaoCategories);
      renderTempStandTerminalAccessChoices(Array.isArray(st.allowedTerminals) ? st.allowedTerminals : []);
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
      const runwayLineupInputCw = document.getElementById('runwayLineupDistM_CW');
      const runwayLineupInputCcw = document.getElementById('runwayLineupDistM_CCW');
      if (tw.pathType === 'runway') {
        if (runwayLineupInputCw) runwayLineupInputCw.value = String(getRunwayLineupDistMByDirection(tw, 'clockwise'));
        if (runwayLineupInputCcw) runwayLineupInputCcw.value = String(getRunwayLineupDistMByDirection(tw, 'counter_clockwise'));
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
      const kindSel = document.getElementById('taxiwayPathTypeKind');
      if (kindSel) {
        const ptk = tw.pathType || 'taxiway';
        kindSel.value = (ptk === 'general_queue_taxiway') ? 'queue' : 'normal';
      }
      syncPathPavementRadiosToValue(pathPavementResolvedForTaxiway(tw));
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
        const twKindIdle = document.getElementById('taxiwayPathTypeKind');
        if (twKindIdle && ptx === 'taxiway') twKindIdle.value = 'normal';
        syncPathPavementRadiosToValue(pathPavementDefaultForPathType(ptx));
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
        const taxiwayTypeWrapIdle = document.getElementById('taxiwayTypeWrap');
        if (taxiwayTypeWrapIdle) taxiwayTypeWrapIdle.style.display = 'none';
      }
      const selIsTerminal = state.selectedObject && state.selectedObject.type === 'terminal';
      if (!selIsTerminal) {
        const buildingTypeSel = document.getElementById('buildingType');
        if (buildingTypeSel) {
          buildingTypeSel.innerHTML = getBuildingTypeOptionsHtml(BUILDING_TYPE_DEFAULT);
          buildingTypeSel.value = BUILDING_TYPE_DEFAULT;
        }
        const terminalNameInput = document.getElementById('terminalName');
        if (terminalNameInput && rm === 'terminal') terminalNameInput.value = getDefaultBuildingNameForType(BUILDING_TYPE_DEFAULT, null);
      }
      const skipStandPanelIdleReset = state.selectedObject && (
        state.selectedObject.type === 'pbb' ||
        state.selectedObject.type === 'remote' ||
        state.selectedObject.type === 'tempStand'
      );
      if (!skipStandPanelIdleReset) {
        applyIcaoCategoriesToHost('standIcaoCategories', ['C']);
        syncStandConstraintVisibility('stand');
        renderAircraftConstraintChoices('standAircraftAccess', aircraftTypeIdsForIcaoLetters(['C']), ['C']);
        applyIcaoCategoriesToHost('remoteIcaoCategories', ['C']);
        syncStandConstraintVisibility('remote');
        renderAircraftConstraintChoices('remoteAircraftAccess', aircraftTypeIdsForIcaoLetters(['C']), ['C']);
        renderRemoteTerminalAccessChoices([]);
        applyIcaoCategoriesToHost('tempStandIcaoCategories', ['C']);
        syncStandConstraintVisibility('tempStand');
        renderAircraftConstraintChoices('tempStandAircraftAccess', aircraftTypeIdsForIcaoLetters(['C']), ['C']);
        renderTempStandTerminalAccessChoices([]);
      }
      const tempStandNameInput = document.getElementById('tempStandName');
      if (tempStandNameInput && rm === 'tempStand' && !(state.selectedObject && state.selectedObject.type === 'tempStand')) tempStandNameInput.value = '';
      const standNameInputIdle = document.getElementById('standName');
      if (standNameInputIdle && rm === 'pbb' && !(state.selectedObject && state.selectedObject.type === 'pbb')) standNameInputIdle.value = '';
      const remoteNameInputIdle = document.getElementById('remoteName');
      if (remoteNameInputIdle && rm === 'remote' && !(state.selectedObject && state.selectedObject.type === 'remote')) remoteNameInputIdle.value = '';
      const taxiwayNameInputIdle = document.getElementById('taxiwayName');
      if (taxiwayNameInputIdle && isPathLayoutMode(rm) && !(state.selectedObject && state.selectedObject.type === 'taxiway')) taxiwayNameInputIdle.value = '';
      const apronLinkNameInput = document.getElementById('apronLinkName');
      if (apronLinkNameInput && rm === 'apronTaxiway') apronLinkNameInput.value = '';
      const edgeNameInput = document.getElementById('edgeName');
      if (edgeNameInput && rm === 'edge') edgeNameInput.value = '';
      const holdingPointNameInput = document.getElementById('holdingPointName');
      if (holdingPointNameInput && rm === 'holdingPoint') holdingPointNameInput.value = getDefaultHoldingPointLabel();
    }
    syncDrawToggleButton('btnTaxiwayDraw', !!state.taxiwayDrawingId);
    syncDrawToggleButton('btnApronLinkDraw', !!state.apronLinkDrawing);
    syncDrawToggleButton('btnPbbDraw', !!state.pbbDrawing);
    syncDrawToggleButton('btnRemoteDraw', !!state.remoteDrawing);
    syncDrawToggleButton('btnTempStandDraw', !!state.tempStandDrawing);
    syncDrawToggleButton('btnHoldingPointDraw', !!state.holdingPointDrawing);
    syncDrawToggleButton('btnMarkerDraw', !!state.markerDrawing);
    renderObjectList();
  }

  function syncStateFromPanel() {
    var el = function(id) { return document.getElementById(id); };
    if (el('gridCellSize')) CELL_SIZE = Math.max(5, Number(el('gridCellSize').value) || 5);
    if (el('gridCols')) GRID_COLS = Math.max(5, Math.min(1000, parseInt(el('gridCols').value, 10) || 200));
    if (el('gridRows')) GRID_ROWS = Math.max(5, Math.min(1000, parseInt(el('gridRows').value, 10) || 200));
    if (state.layoutImageOverlay) {
      state.layoutImageOverlay.opacity = clampLayoutImageOpacity(el('gridLayoutImageOpacity') ? el('gridLayoutImageOpacity').value : state.layoutImageOverlay.opacity);
      state.layoutImageOverlay.widthM = clampLayoutImageSize(el('gridLayoutImageWidthM') ? el('gridLayoutImageWidthM').value : state.layoutImageOverlay.widthM, state.layoutImageOverlay.widthM);
      state.layoutImageOverlay.heightM = clampLayoutImageSize(el('gridLayoutImageHeightM') ? el('gridLayoutImageHeightM').value : state.layoutImageOverlay.heightM, state.layoutImageOverlay.heightM);
      state.layoutImageOverlay.topLeftCol = clampLayoutImagePoint(el('gridLayoutImageCol') ? el('gridLayoutImageCol').value : state.layoutImageOverlay.topLeftCol, state.layoutImageOverlay.topLeftCol);
      state.layoutImageOverlay.topLeftRow = clampLayoutImagePoint(el('gridLayoutImageRow') ? el('gridLayoutImageRow').value : state.layoutImageOverlay.topLeftRow, state.layoutImageOverlay.topLeftRow);
    }
    var t = getCurrentTerminal();
    if (t) {
      if (el('terminalName')) {
        const rawTn = (el('terminalName').value || '').trim();
        if (rawTn && findDuplicateLayoutName('terminal', t.id, rawTn)) {
          alertDuplicateLayoutName();
          el('terminalName').value = t.name || '';
        } else {
          t.name = rawTn || t.name;
        }
      }
      if (el('buildingType')) t.buildingType = normalizeBuildingType(el('buildingType').value || t.buildingType);
      if (el('terminalFloors')) t.floors = Math.max(1, parseInt(el('terminalFloors').value, 10) || 1);
      if (el('terminalFloorToFloor')) t.floorToFloor = Math.max(0.5, Number(el('terminalFloorToFloor').value) || 4);
      t.floorHeight = (t.floors || 1) * (t.floorToFloor || 4);
      if (el('terminalDepartureCapacity')) t.departureCapacity = Math.max(0, parseInt(el('terminalDepartureCapacity').value, 10) || 0);
      if (el('terminalArrivalCapacity')) t.arrivalCapacity = Math.max(0, parseInt(el('terminalArrivalCapacity').value, 10) || 0);
    }
    if (state.selectedObject && state.selectedObject.type === 'pbb') {
      var pbb = state.selectedObject.obj;
      if (el('standName')) {
        const rawSn = (el('standName').value || '').trim();
        if (rawSn && findDuplicateLayoutName('pbb', pbb.id, rawSn)) {
          alertDuplicateLayoutName();
          el('standName').value = pbb.name || '';
        } else {
          pbb.name = rawSn;
        }
      }
      applyUnifiedStandConstraintFromPanelToObject(pbb, 'standIcaoCategories', 'standAircraftAccess');
    }
    if (state.selectedObject && state.selectedObject.type === 'remote') {
      var st = state.selectedObject.obj;
      if (el('remoteName')) {
        const rawRn = (el('remoteName').value || '').trim();
        if (rawRn && findDuplicateLayoutName('remote', st.id, rawRn)) {
          alertDuplicateLayoutName();
          el('remoteName').value = st.name || '';
        } else {
          st.name = rawRn;
        }
      }
      applyUnifiedStandConstraintFromPanelToObject(st, 'remoteIcaoCategories', 'remoteAircraftAccess');
      const accWrap = document.getElementById('remoteTerminalAccess');
      if (accWrap) {
        const checks = accWrap.querySelectorAll('.remote-term-check');
        const allowed = [];
        checks.forEach(function(ch) {
          if (ch.checked) {
            const id = ch.getAttribute('data-item-id');
            if (id) allowed.push(id);
          }
        });
        st.allowedTerminals = allowed;
      }
    }
    if (state.selectedObject && state.selectedObject.type === 'tempStand') {
      var tst = state.selectedObject.obj;
      if (el('tempStandName')) {
        const rawTn = (el('tempStandName').value || '').trim();
        if (rawTn && findDuplicateLayoutName('tempStand', tst.id, rawTn)) {
          alertDuplicateLayoutName();
          el('tempStandName').value = tst.name || '';
