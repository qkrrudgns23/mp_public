        terminals: JSON.parse(JSON.stringify(state.terminals || [])),
        pbbStands: JSON.parse(JSON.stringify(state.pbbStands || [])),
        remoteStands: JSON.parse(JSON.stringify(state.remoteStands || [])),
        holdingPoints: JSON.parse(JSON.stringify(state.holdingPoints || [])),
        taxiways: JSON.parse(JSON.stringify(state.taxiways || [])),
        apronLinks: JSON.parse(JSON.stringify(state.apronLinks || [])),
        layoutImageOverlay: snapshot,
        layoutEdgeNames: JSON.parse(JSON.stringify(state.layoutEdgeNames || {})),
        directionModes: JSON.parse(JSON.stringify(state.directionModes || [])),
        flights: cloneFlightsWithoutPathPolylineCache(state.flights)
      });
      if (undoStack.length > maxUndoLevels) undoStack.shift();
      syncPanelFromState();
      invalidateGridUnderlay();
      draw();
    });
  }
  const gridLayoutImageFileEl = document.getElementById('gridLayoutImageFile');
  if (gridLayoutImageFileEl) {
    gridLayoutImageFileEl.addEventListener('change', function() {
      const file = this.files && this.files[0];
      if (!file) return;
      const fileType = String(file.type || '').toLowerCase();
      const fileName = String(file.name || 'Layout image');
      const accepted = fileType === 'image/png' || fileType === 'image/jpeg' || fileType === 'image/svg+xml' ||
        /\.(png|jpe?g|svg)$/i.test(fileName);
      if (!accepted) {
        alert('Only PNG, JPG, JPEG, and SVG files are supported.');
        this.value = '';
        return;
      }
      const reader = new FileReader();
      reader.onload = function(ev) {
        const dataUrl = ev && ev.target ? String(ev.target.result || '') : '';
        if (!dataUrl) return;
        const img = new Image();
        img.onload = function() {
          const widthM = state.layoutImageOverlay ? clampLayoutImageSize(state.layoutImageOverlay.widthM, GRID_LAYOUT_IMAGE_DEFAULTS.widthM) : GRID_LAYOUT_IMAGE_DEFAULTS.widthM;
          const aspect = (img.naturalWidth > 0 && img.naturalHeight > 0)
            ? (img.naturalHeight / img.naturalWidth)
            : (GRID_LAYOUT_IMAGE_DEFAULTS.heightM / Math.max(GRID_LAYOUT_IMAGE_DEFAULTS.widthM, 1e-9));
          const heightM = state.layoutImageOverlay
            ? clampLayoutImageSize(state.layoutImageOverlay.heightM, Math.max(1, widthM * aspect))
            : Math.max(1, widthM * aspect);
          pushUndo();
          state.layoutImageOverlay = normalizeLayoutImageOverlay({
            name: fileName,
            type: fileType || 'image/png',
            dataUrl: dataUrl,
            opacity: state.layoutImageOverlay ? state.layoutImageOverlay.opacity : GRID_LAYOUT_IMAGE_DEFAULTS.opacity,
            widthM: widthM,
            heightM: heightM,
            originalWidthPx: img.naturalWidth || widthM,
            originalHeightPx: img.naturalHeight || heightM,
            topLeftCol: state.layoutImageOverlay ? state.layoutImageOverlay.topLeftCol : GRID_LAYOUT_IMAGE_DEFAULTS.topLeftCol,
            topLeftRow: state.layoutImageOverlay ? state.layoutImageOverlay.topLeftRow : GRID_LAYOUT_IMAGE_DEFAULTS.topLeftRow
          });
          syncLayoutImageBitmap();
          syncPanelFromState();
          draw();
        };
        img.onerror = function() {
          alert('Failed to read the selected layout image.');
          gridLayoutImageFileEl.value = '';
        };
        img.src = dataUrl;
      };
      reader.readAsDataURL(file);
    });
  }
  const clearGridLayoutImageBtn = document.getElementById('btnClearGridLayoutImage');
  if (clearGridLayoutImageBtn) {
    clearGridLayoutImageBtn.addEventListener('click', function() {
      if (!state.layoutImageOverlay) return;
      pushUndo();
      state.layoutImageOverlay = null;
      layoutImageBitmap = null;
      layoutImageBitmapSrc = '';
      if (gridLayoutImageFileEl) gridLayoutImageFileEl.value = '';
      syncPanelFromState();
      draw();
    });
  }
  commitGridLayoutImageNumericChange('gridLayoutImageOpacity', function(input) {
    state.layoutImageOverlay.opacity = clampLayoutImageOpacity(input.value);
  });
  commitGridLayoutImageNumericChange('gridLayoutImageWidthM', function(input) {
    applyLayoutImageWidthByAspect(input.value);
  });
  commitGridLayoutImageNumericChange('gridLayoutImageHeightM', function(input) {
    applyLayoutImageHeightByAspect(input.value);
  });
  commitGridLayoutImageNumericChange('gridLayoutImageCol', function(input) {
    state.layoutImageOverlay.topLeftCol = clampLayoutImagePoint(input.value, state.layoutImageOverlay.topLeftCol);
  });
  commitGridLayoutImageNumericChange('gridLayoutImageRow', function(input) {
    state.layoutImageOverlay.topLeftRow = clampLayoutImagePoint(input.value, state.layoutImageOverlay.topLeftRow);
  });

  document.getElementById('terminalName').addEventListener('change', function() {
    const t = getCurrentTerminal();
    if (t) {
      const raw = (this.value || '').trim();
      if (raw && findDuplicateLayoutName('terminal', t.id, raw)) {
        alertDuplicateLayoutName();
        this.value = t.name || '';
        return;
      }
      t.name = raw || t.name;
      draw();
      updateObjectInfo();
      if (typeof markGlobalUpdateStale === 'function') markGlobalUpdateStale();
    }
  });
  const buildingTypeInput = document.getElementById('buildingType');
  if (buildingTypeInput) {
    buildingTypeInput.addEventListener('change', function() {
      const nextType = normalizeBuildingType(this.value || BUILDING_TYPE_DEFAULT);
      const t = getCurrentTerminal();
      const nameInput = document.getElementById('terminalName');


      const nextDefaultName = getDefaultBuildingNameForType(nextType, t ? t.id : null);
      if (t) {
        t.buildingType = nextType;
        if (findDuplicateLayoutName('terminal', t.id, nextDefaultName)) {
          alertDuplicateLayoutName();
          if (nameInput) nameInput.value = t.name || '';
        } else {
          t.name = nextDefaultName;
          if (nameInput) nameInput.value = nextDefaultName;
        }
      } else if (nameInput) {
        nameInput.value = nextDefaultName;
      }
      updateObjectInfo();
      renderObjectList();
      draw();
      if (typeof update3DScene === 'function') update3DScene();
      if (typeof markGlobalUpdateStale === 'function') markGlobalUpdateStale();
    });
  }
  function recomputeTerminalFloorHeight() {
    const t = getCurrentTerminal();
    if (!t) return;
    const floorsInput = document.getElementById('terminalFloors');
    const f2fInput = document.getElementById('terminalFloorToFloor');
    const totalInput = document.getElementById('terminalFloorHeight');
    let floors = floorsInput ? parseInt(floorsInput.value, 10) : t.floors;
    let f2f = f2fInput ? Number(f2fInput.value) : t.floorToFloor;
    floors = Math.max(1, floors || 1);
    f2f = Math.max(0.5, f2f || 4);
    const totalH = floors * f2f;
    t.floors = floors;
    t.floorToFloor = f2f;
    t.floorHeight = totalH;
    if (floorsInput) floorsInput.value = floors;
    if (f2fInput) f2fInput.value = f2f;
    if (totalInput) totalInput.value = totalH;
    draw();
    updateObjectInfo();
    if (typeof update3DScene === 'function') update3DScene();
  }
  document.getElementById('terminalFloors').addEventListener('change', recomputeTerminalFloorHeight);
  document.getElementById('terminalFloorToFloor').addEventListener('change', recomputeTerminalFloorHeight);
  document.getElementById('terminalDepartureCapacity').addEventListener('change', function() {
    const t = getCurrentTerminal();
    if (t) { t.departureCapacity = Math.max(0, parseInt(this.value, 10) || 0); updateObjectInfo(); }
  });
  document.getElementById('terminalArrivalCapacity').addEventListener('change', function() {
    const t = getCurrentTerminal();
    if (t) { t.arrivalCapacity = Math.max(0, parseInt(this.value, 10) || 0); updateObjectInfo(); }
  });

  document.getElementById('standName').addEventListener('change', function() {
    if (state.selectedObject && state.selectedObject.type === 'pbb') {
      const pbb = state.selectedObject.obj;
      const raw = (this.value || '').trim();
      if (raw && findDuplicateLayoutName('pbb', pbb.id, raw)) {
        alertDuplicateLayoutName();
        this.value = pbb.name || '';
        return;
      }
      pbb.name = raw;
      updateObjectInfo();
      renderObjectList();
      draw();
    }
  });
  document.getElementById('standCategory').addEventListener('change', function() {
    const val = this.value || 'C';
    if (state.selectedObject && state.selectedObject.type === 'pbb') {
      state.selectedObject.obj.category = val;
      rebuildPbbBridgeGeometry(state.selectedObject.obj);
      updateObjectInfo();
      renderObjectList();
      draw();
      if (typeof update3DScene === 'function') update3DScene();
    }
  });
  const standCategoryModeEl = document.getElementById('standCategoryMode');
  if (standCategoryModeEl) {
    standCategoryModeEl.addEventListener('change', function() {
      syncStandConstraintVisibility('stand', this.value);
      if (state.selectedObject && state.selectedObject.type === 'pbb') {
        state.selectedObject.obj.categoryMode = normalizeStandCategoryMode(this.value, _pbbTier.defaultCategoryMode || 'icao');
        updateObjectInfo();
        renderObjectList();
        draw();
      }
    });
  }
  const pbbLengthInputEl = document.getElementById('pbbLength');
  if (pbbLengthInputEl) {
    pbbLengthInputEl.addEventListener('change', function() {
      const requested = Number(this.value);
      const nextLen = (isFinite(requested) && requested > 0) ? requested : 15;
      this.value = String(Math.max(1, Math.round(nextLen)));
      if (state.selectedObject && state.selectedObject.type === 'pbb') {
        const pbb = state.selectedObject.obj;
        setPbbGeometryFromAngleLength(pbb, getPbbAngleDeg(pbb), nextLen, true);
        updateObjectInfo();
        renderObjectList();
        draw();
        if (typeof update3DScene === 'function') update3DScene();
      }
    });
  }
  const standAngleInputEl = document.getElementById('standAngle');
  if (standAngleInputEl) {
    standAngleInputEl.addEventListener('change', function() {
      const nextDeg = normalizeAngleDeg(this.value);
      this.value = String(Math.round(nextDeg));
      if (state.selectedObject && state.selectedObject.type === 'pbb') {
        const pbb = state.selectedObject.obj;
        setPbbGeometryFromAngleLength(pbb, nextDeg, getPbbLengthMeters(pbb), true);
        updateObjectInfo();
        renderObjectList();
        draw();
        if (typeof update3DScene === 'function') update3DScene();
      }
    });
  }
  const pbbBridgeCountInputEl = document.getElementById('pbbBridgeCount');
  if (pbbBridgeCountInputEl) {
    pbbBridgeCountInputEl.addEventListener('change', function() {
      const nextCount = Math.max(1, Math.min(8, parseInt(this.value, 10) || 1));
      this.value = String(nextCount);
      if (state.selectedObject && state.selectedObject.type === 'pbb') {
        const pbb = state.selectedObject.obj;
        pbb.pbbCount = nextCount;
        delete pbb.pbbBridges;
        rebuildPbbBridgeGeometry(pbb);
        updateObjectInfo();
        renderObjectList();
        draw();
        if (typeof update3DScene === 'function') update3DScene();
      }
    });
  }
  const standAircraftAccessEl = document.getElementById('standAircraftAccess');
  if (standAircraftAccessEl) {
    standAircraftAccessEl.addEventListener('change', function(ev) {
      const target = ev.target;
      if (!target || !target.classList.contains('aircraft-type-check')) return;
      syncChoiceChipStates(standAircraftAccessEl);
      if (!state.selectedObject || state.selectedObject.type !== 'pbb') return;
      state.selectedObject.obj.allowedAircraftTypes = readCheckedDataItemIds(standAircraftAccessEl, '.aircraft-type-check');
      updateObjectInfo();
      renderObjectList();
      draw();
    });
  }

  const remoteNameInput = document.getElementById('remoteName');
  if (remoteNameInput) {
    remoteNameInput.addEventListener('change', function() {
      if (state.selectedObject && state.selectedObject.type === 'remote') {
        const st = state.selectedObject.obj;
        const raw = (this.value || '').trim();
        if (raw && findDuplicateLayoutName('remote', st.id, raw)) {
          alertDuplicateLayoutName();
          this.value = st.name || '';
          return;
        }
        st.name = raw;
        updateObjectInfo();
        renderObjectList();
        draw();
        if (typeof update3DScene === 'function') update3DScene();
      }
    });
  }
  const holdingPointNameInput = document.getElementById('holdingPointName');
  if (holdingPointNameInput) {
    holdingPointNameInput.addEventListener('change', function() {
      if (state.selectedObject && state.selectedObject.type === 'holdingPoint') {
        const hp = state.selectedObject.obj;
        const raw = (this.value || '').trim();
        if (raw && findDuplicateLayoutName('holdingPoint', hp.id, raw)) {
          alertDuplicateLayoutName();
          this.value = hp.name || '';
          return;
        }
        hp.name = raw;
        updateObjectInfo();
        renderObjectList();
        draw();
      }
    });
  }
  const remoteCategorySelect = document.getElementById('remoteCategory');
  if (remoteCategorySelect) {
    remoteCategorySelect.addEventListener('change', function() {
      const val = this.value || 'C';
      if (state.selectedObject && state.selectedObject.type === 'remote') {
        state.selectedObject.obj.category = val;
        updateObjectInfo();
        renderObjectList();
        draw();
        if (typeof update3DScene === 'function') update3DScene();
      }
    });
  }
  const remoteCategoryModeEl = document.getElementById('remoteCategoryMode');
  if (remoteCategoryModeEl) {
    remoteCategoryModeEl.addEventListener('change', function() {
      syncStandConstraintVisibility('remote', this.value);
      if (state.selectedObject && state.selectedObject.type === 'remote') {
        state.selectedObject.obj.categoryMode = normalizeStandCategoryMode(this.value, _remoteTier.defaultCategoryMode || 'icao');
        updateObjectInfo();
        renderObjectList();
        draw();
      }
    });
  }
  const remoteAngleInput = document.getElementById('remoteAngle');
  if (remoteAngleInput) {
    remoteAngleInput.addEventListener('change', function() {
      const nextDeg = normalizeAngleDeg(this.value);
      this.value = String(Math.round(nextDeg));
      if (state.selectedObject && state.selectedObject.type === 'remote') {
        state.selectedObject.obj.angleDeg = nextDeg;
        updateObjectInfo();
        renderObjectList();
        draw();
        if (typeof update3DScene === 'function') update3DScene();
      }
    });
  }

  const remoteTerminalAccessEl = document.getElementById('remoteTerminalAccess');
  if (remoteTerminalAccessEl) {
    remoteTerminalAccessEl.addEventListener('change', function(ev) {
      const target = ev.target;
      if (!target || !target.classList.contains('remote-term-check')) return;
      syncChoiceChipStates(remoteTerminalAccessEl);
      if (!state.selectedObject || state.selectedObject.type !== 'remote') return;
      const st = state.selectedObject.obj;
      const checks = remoteTerminalAccessEl.querySelectorAll('.remote-term-check');
      const allowed = [];
      checks.forEach(function(ch) {
        if (ch.checked) {
          const id = ch.getAttribute('data-item-id');
          if (id) allowed.push(id);
        }
      });
      st.allowedTerminals = allowed;
      if (typeof syncPanelFromState === 'function') syncPanelFromState();
      updateObjectInfo();
      renderObjectList();
      draw();
    });
  }
  const remoteAircraftAccessEl = document.getElementById('remoteAircraftAccess');
  if (remoteAircraftAccessEl) {
    remoteAircraftAccessEl.addEventListener('change', function(ev) {
      const target = ev.target;
      if (!target || !target.classList.contains('aircraft-type-check')) return;
      syncChoiceChipStates(remoteAircraftAccessEl);
      if (!state.selectedObject || state.selectedObject.type !== 'remote') return;
      state.selectedObject.obj.allowedAircraftTypes = readCheckedDataItemIds(remoteAircraftAccessEl, '.aircraft-type-check');
      updateObjectInfo();
      renderObjectList();
      draw();
    });
  }

  document.getElementById('taxiwayName').addEventListener('change', function() {
    if (state.selectedObject && state.selectedObject.type === 'taxiway') {
      const tw = state.selectedObject.obj;
      const raw = (this.value || '').trim();
      if (raw && findDuplicateLayoutName('taxiway', tw.id, raw)) {
        alertDuplicateLayoutName();
        this.value = tw.name || '';
        return;
      }
      tw.name = raw;
      updateObjectInfo();
      renderObjectList();
      draw();
    }
  });
  const apronLinkNameInputEl = document.getElementById('apronLinkName');
  if (apronLinkNameInputEl) {
    apronLinkNameInputEl.addEventListener('change', function() {
      if (state.selectedObject && state.selectedObject.type === 'apronLink') {
        const lk = state.selectedObject.obj;
        const rawTrim = (this.value || '').trim();
        const candidate = rawTrim || getApronLinkDefaultName(lk.id);
        if (findDuplicateLayoutName('apronLink', lk.id, candidate)) {
          alertDuplicateLayoutName();
          this.value = getApronLinkDisplayName(lk);
          return;
        }
        lk.name = rawTrim;
        this.value = getApronLinkDisplayName(lk);
        updateObjectInfo();
        renderObjectList();
        draw();
      }
    });
  }
  const edgeNameInputEl = document.getElementById('edgeName');
  if (edgeNameInputEl) {
    edgeNameInputEl.addEventListener('change', function() {
      if (state.selectedObject && state.selectedObject.type === 'layoutEdge') {
        const ed = state.selectedObject.obj;
        const rawTrim = (this.value || '').trim();
        const candidate = rawTrim || getLayoutEdgeDefaultName(ed);
        if (findDuplicateLayoutName('layoutEdge', ed.id, candidate)) {
          alertDuplicateLayoutName();
          this.value = getLayoutEdgeDisplayName(ed);
          return;
        }
        state.layoutEdgeNames[ed.id] = candidate;
        ed.name = candidate;
        this.value = candidate;
        updateObjectInfo();
        renderObjectList();
        draw();
      }
    });
  }
  document.getElementById('taxiwayWidth').addEventListener('change', function() {
    if (state.selectedObject && state.selectedObject.type === 'taxiway') {
      const tw = state.selectedObject.obj;
      const baseWidth = tw.pathType === 'runway'
        ? RUNWAY_PATH_DEFAULT_WIDTH
        : (tw.pathType === 'runway_exit' ? RUNWAY_EXIT_DEFAULT_WIDTH : TAXIWAY_DEFAULT_WIDTH);
