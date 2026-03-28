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
      const val = Number(this.value);
      tw.width = clampTaxiwayWidthM(tw.pathType || 'taxiway', val, baseWidth);
      this.value = tw.width;
      updateObjectInfo();
      draw();
      if (scene3d) update3DScene();
    }
  });
  const avgVelInputEl = document.getElementById('taxiwayAvgMoveVelocity');
  if (avgVelInputEl) avgVelInputEl.addEventListener('change', function() {
    if (state.selectedObject && state.selectedObject.type === 'taxiway') {
      const tw = state.selectedObject.obj;
      const val = Number(this.value);
      const v =
        (typeof val === 'number' && isFinite(val) && val > 0)
          ? Math.max(1, Math.min(50, val))
          : 10;
      tw.avgMoveVelocity = v;
      this.value = v;
      updateObjectInfo();
      renderObjectList();
      draw();
      if (typeof update3DScene === 'function') update3DScene();
    }
  });
  document.getElementById('taxiwayMaxExitVel').addEventListener('change', function() {
    if (state.selectedObject && state.selectedObject.type === 'taxiway') {
      const tw = state.selectedObject.obj;
      const val = Number(this.value);
      if (tw.pathType === 'runway_exit') {
        tw.maxExitVelocity = isFinite(val) && val > 0 ? val : null;
        if (typeof tw.minExitVelocity === 'number' && isFinite(tw.minExitVelocity) && tw.maxExitVelocity != null && tw.minExitVelocity > tw.maxExitVelocity) {
          tw.minExitVelocity = tw.maxExitVelocity;
        }
      } else {
        delete tw.maxExitVelocity;
      }
      if (isFinite(val) && val > 0) this.value = val; else this.value = tw.maxExitVelocity != null ? tw.maxExitVelocity : '';
      updateObjectInfo();
      renderObjectList();
      draw();
      if (scene3d) update3DScene();
    }
  });
  const minExitEl = document.getElementById('taxiwayMinExitVel');
  if (minExitEl) {
    minExitEl.addEventListener('change', function() {
      if (state.selectedObject && state.selectedObject.type === 'taxiway') {
        const tw = state.selectedObject.obj;
        const val = Number(this.value);
        if (tw.pathType === 'runway_exit') {
          let v = isFinite(val) && val > 0 ? val : 15;
          if (typeof tw.maxExitVelocity === 'number' && isFinite(tw.maxExitVelocity) && v > tw.maxExitVelocity) v = tw.maxExitVelocity;
          tw.minExitVelocity = v;
          this.value = v;
        } else {
          delete tw.minExitVelocity;
        }
        updateObjectInfo();
        renderObjectList();
        draw();
        if (scene3d) update3DScene();
      }
    });
  }
  const runwayExitAllowedDirectionEl = document.getElementById('runwayExitAllowedDirection');
  if (runwayExitAllowedDirectionEl) {
    runwayExitAllowedDirectionEl.addEventListener('change', function(ev) {
      const target = ev.target;
      if (!target || !target.classList.contains('runway-exit-dir-check')) return;
      syncChoiceChipStates(runwayExitAllowedDirectionEl);
      if (!(state.selectedObject && state.selectedObject.type === 'taxiway')) return;
      const tw = state.selectedObject.obj;
      if (!tw || tw.pathType !== 'runway_exit') return;
      tw.allowedRwDirections = getRunwayExitAllowedDirectionsFromPanel();
        updateObjectInfo();
        renderObjectList();
        if (typeof redrawLayoutAfterEdit === 'function') redrawLayoutAfterEdit();
        else if (typeof updateAllFlightPaths === 'function') updateAllFlightPaths();
        else draw();
      });
  }
  document.getElementById('taxiwayDirectionMode').addEventListener('change', function() {
    if (state.selectedObject && state.selectedObject.type === 'taxiway') {
      const tw = state.selectedObject.obj;
      const v = this.value || '';
      tw.direction = (tw.pathType === 'runway') ? ((v === 'counter_clockwise') ? 'counter_clockwise' : 'clockwise') : (v || 'both');
      updateObjectInfo();
      if (typeof markGlobalUpdateStale === 'function') markGlobalUpdateStale();
      draw();
      if (typeof update3DScene === 'function') update3DScene();
    }
  });
  const runwayMinArrVelEl = document.getElementById('runwayMinArrVelocity');
  if (runwayMinArrVelEl) {
    runwayMinArrVelEl.addEventListener('change', function() {
      if (state.selectedObject && state.selectedObject.type === 'taxiway') {
        const tw = state.selectedObject.obj;
        if (tw.pathType !== 'runway') return;
        const val = Number(this.value);
        const v = (typeof val === 'number' && isFinite(val) && val > 0) ? Math.max(1, Math.min(150, val)) : 15;
        tw.minArrVelocity = v;
        this.value = v;
        updateObjectInfo();
        renderObjectList();
        if (typeof markGlobalUpdateStale === 'function') markGlobalUpdateStale();
        draw();
      }
    });
  }
  const runwayLineupEl = document.getElementById('runwayLineupDistM');
  if (runwayLineupEl) {
    runwayLineupEl.addEventListener('change', function() {
      if (state.selectedObject && state.selectedObject.type === 'taxiway') {
        const tw = state.selectedObject.obj;
        if (tw.pathType !== 'runway') return;
        const val = Number(this.value);
        const v = (typeof val === 'number' && isFinite(val) && val >= 0) ? val : 0;
        tw.lineupDistM = v;
        this.value = String(v);
        updateObjectInfo();
        if (typeof redrawLayoutAfterEdit === 'function') redrawLayoutAfterEdit();
        else if (typeof updateAllFlightPaths === 'function') updateAllFlightPaths(); else draw();
      }
    });
  }
  [
    ['runwayStartDisplacedThresholdM', 'startDisplacedThresholdM', function(tw) { return getEffectiveRunwayStartDisplacedThresholdM(tw); }],
    ['runwayStartBlastPadM', 'startBlastPadM', function(tw) { return getEffectiveRunwayStartBlastPadM(tw); }],
    ['runwayEndDisplacedThresholdM', 'endDisplacedThresholdM', function(tw) { return getEffectiveRunwayEndDisplacedThresholdM(tw); }],
    ['runwayEndBlastPadM', 'endBlastPadM', function(tw) { return getEffectiveRunwayEndBlastPadM(tw); }]
  ].forEach(function(item) {
    const el = document.getElementById(item[0]);
    if (!el) return;
    el.addEventListener('change', function() {
      if (state.selectedObject && state.selectedObject.type === 'taxiway') {
        const tw = state.selectedObject.obj;
        if (tw.pathType !== 'runway') return;
        const val = Number(this.value);
        const v = (typeof val === 'number' && isFinite(val) && val >= 0) ? val : item[2](tw);
        tw[item[1]] = v;
        this.value = String(v);
        updateObjectInfo();
        draw();
        if (typeof markGlobalUpdateStale === 'function') markGlobalUpdateStale();
