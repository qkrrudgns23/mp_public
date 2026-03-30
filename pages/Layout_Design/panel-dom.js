    });
    const tokenRunwaySel = document.getElementById('tokenRunwaySelect');
    const tokenTerminalSel = document.getElementById('tokenTerminalSelect');
    if (tokenRunwaySel) tokenRunwaySel.addEventListener('change', function() {
      if (!state.selectedObject || state.selectedObject.type !== 'flight') return;
      const f = state.selectedObject.obj;
      if (!f.token) f.token = { nodes: TOKEN_NODE_ORDER.slice(), runwayId: null, apronId: null, terminalId: null };
      f.token.runwayId = this.value || null;
      rebuildSelectedFlightTimeline();
    });
    if (tokenTerminalSel) tokenTerminalSel.addEventListener('change', function() {
      if (!state.selectedObject || state.selectedObject.type !== 'flight') return;
      const f = state.selectedObject.obj;
      if (!f.token) f.token = { nodes: TOKEN_NODE_ORDER.slice(), runwayId: null, apronId: null, terminalId: null };
      f.token.terminalId = this.value || null;
      rebuildSelectedFlightTimeline();
    });
    const flightSubtabButtons = document.querySelectorAll('.flight-subtab');
    const flightPaneSchedule = document.getElementById('flightPaneSchedule');
    const flightPaneConfig = document.getElementById('flightPaneConfig');
    if (flightSubtabButtons && flightPaneSchedule && flightPaneConfig) {
      flightSubtabButtons.forEach(btn => {
        btn.addEventListener('click', function() {
          const target = this.getAttribute('data-flight-subtab') || 'schedule';
          flightSubtabButtons.forEach(b => b.classList.remove('active'));
          this.classList.add('active');
          if (target === 'config') {
            flightPaneSchedule.style.display = 'none';
            flightPaneConfig.style.display = 'block';
          } else {
            flightPaneSchedule.style.display = 'block';
            flightPaneConfig.style.display = 'none';
          }
        });
      });
    }
    if (addBtn) {
      addBtn.addEventListener('click', function() {
        const networkErrors = validateNetworkForFlights();
        if (networkErrors.length) {
          updateFlightError(networkErrors);
          alert('Flightcannot be created:\\n' + networkErrors.join('\\n'));
          return;
        }
        let timeStr = (document.getElementById('flightTime').value || '').trim();
        if (!timeStr) {
          const defMin = getDefaultSibtMinutes();
          timeStr = formatMinutesToHHMMSS(defMin);
          if (timeInputEl) timeInputEl.value = timeStr;
        }
        const timeMin = parseTimeToMinutes(timeStr);
        const aircraftType = (document.getElementById('flightAircraftType').value || 'A320').trim();
        const code = getCodeForAircraft(aircraftType);
        const reg = (document.getElementById('flightReg').value || '').trim();
        let airlineCode = (document.getElementById('flightAirlineCode') && document.getElementById('flightAirlineCode').value || '').trim();
        let flightNumber = (document.getElementById('flightFlightNumber') && document.getElementById('flightFlightNumber').value || '').trim();
        if (!airlineCode) airlineCode = randomAirlineCode();
        if (!flightNumber) flightNumber = randomFlightNumber(airlineCode);
        let dwellMin = parseFloat(document.getElementById('flightDwell').value);
        let minDwellMin = parseFloat(document.getElementById('flightMinDwell').value);
        dwellMin = (typeof dwellMin === 'number' && !isNaN(dwellMin) && dwellMin >= 0) ? dwellMin : 0;
        minDwellMin = (typeof minDwellMin === 'number' && !isNaN(minDwellMin) && minDwellMin >= 0) ? minDwellMin : 0;
        dwellMin = Math.max(SCHED_DWELL_FLOOR_MIN, dwellMin);
        minDwellMin = Math.max(SCHED_DWELL_FLOOR_MIN, minDwellMin);
        if (minDwellMin > dwellMin) minDwellMin = dwellMin;
        const arrDep = 'Arr';
        const runwayOptions = getRunwayOptions();
        const defaultRunwayId = runwayOptions.length ? (runwayOptions[0].id || null) : null;
        const f = {
          id: id(),
          arrDep,
          timeMin,
          aircraftType,
          code,
          reg,
          airlineCode,
          flightNumber,
          dwellMin,
          minDwellMin,
          arrRunwayId: defaultRunwayId,
          depRunwayId: defaultRunwayId,
          timeline: null,
          token: {
            nodes: ['runway','taxiway','apron','terminal'],
            runwayId: defaultRunwayId,
            arrRunwayId: defaultRunwayId,
            depRunwayId: defaultRunwayId,
            apronId: null,
            terminalId: null
          }
        };
        computeFlightPath(f, 'arrival');
        computeFlightPath(f, 'departure');
        if (f.noWayArr || f.noWayDep) {
          updateFlightError('NOTE: Available on your network Taxiway / Apron path not found. (Simulation paths may not be drawn.)');
        }
        state.flights.push(f);
        if (typeof syncSimulationPlaybackAfterTimelines === 'function') syncSimulationPlaybackAfterTimelines();
        else if (typeof recomputeSimDuration === 'function') recomputeSimDuration();
        if (typeof markGlobalUpdateStale === 'function') markGlobalUpdateStale();
        var addTouched = f.standId ? [f.standId] : [];
        renderFlightList(false, false, { scheduleMode: 'incremental', dirtyFlightIds: [f.id], touchedStandIds: addTouched });
        if (timeInputEl) {
          const nextDef = getDefaultSibtMinutes();
          timeInputEl.value = formatMinutesToHHMMSS(nextDef);
        }
        updateFlightError('');
      });
    }
    function syncFlightPanelFromSelection() {
      if (!state.selectedObject || state.selectedObject.type !== 'flight') return;
      const f = state.selectedObject.obj;
      if (arrDepEl) arrDepEl.value = 'Arr';
      if (dwellEl) {
        dwellEl.disabled = false;
        dwellEl.value = f.dwellMin || 0;
      }
      if (minDwellEl) {
        minDwellEl.disabled = false;
        minDwellEl.value = f.minDwellMin != null ? f.minDwellMin : 0;
      }
      if (timeInputEl) timeInputEl.value = formatMinutesToHHMMSS(f.timeMin);
      if (aircraftEl) {
        if (f.aircraftType && AIRCRAFT_BY_ID[f.aircraftType]) aircraftEl.value = f.aircraftType;
        else {
          const match = AIRCRAFT_TYPES.find(a => a.icao === (f.code || 'C'));
          aircraftEl.value = match ? match.id : (AIRCRAFT_TYPES[0] && AIRCRAFT_TYPES[0].id) || 'A320';
        }
      }
      if (regEl) regEl.value = f.reg || '';
      const airlineCodeEl = document.getElementById('flightAirlineCode');
      const flightNumberEl = document.getElementById('flightFlightNumber');
      if (airlineCodeEl) airlineCodeEl.value = f.airlineCode || '';
      if (flightNumberEl) flightNumberEl.value = f.flightNumber || '';
      if (!f.token) f.token = { nodes: TOKEN_NODE_ORDER.slice(), runwayId: null, apronId: null, terminalId: null };
      fillTokenSelects(f.code);
      setTokenCheckboxesFromNodes(f.token.nodes);
      if (tokenRunwaySel) tokenRunwaySel.value = f.token.runwayId || '';
      if (tokenTerminalSel) tokenTerminalSel.value = f.token.terminalId || '';
      if (typeof syncFlightAssignStrip === 'function') syncFlightAssignStrip();
    }
    hookSyncFlightPanelFromSelection = syncFlightPanelFromSelection;
    const origSyncPanel = syncPanelFromState;
    syncPanelFromState = function() {
      origSyncPanel();
      if (activeTab === 'flight') syncFlightPanelFromSelection();
    };
    function rebuildSelectedFlightTimeline() {
      if (!state.selectedObject || state.selectedObject.type !== 'flight') return;
      if (typeof markGlobalUpdateStale === 'function') markGlobalUpdateStale();
      const f = state.selectedObject.obj;
      computeFlightPath(f, 'arrival');
      computeFlightPath(f, 'departure');
      const isArr = f.arrDep !== 'Dep';
      if (isArr && f.noWayArr) {
        updateFlightError('no path(No Way): Arrival route not found.');
        f.timeline = null;
        draw();
        return;
      }
      if (!isArr && f.noWayDep) {
        updateFlightError('no path(No Way): Departure route not found.');
        f.timeline = null;
        draw();
        return;
      }
      if (typeof buildFullAirsideTimelineForFlight === 'function') buildFullAirsideTimelineForFlight(f);
      if (!f.timeline || !f.timeline.length) {
        updateFlightError('No valid route found on that network. (After changing settings)');
        return;
      }
      if (typeof syncSimulationPlaybackAfterTimelines === 'function') syncSimulationPlaybackAfterTimelines();
      else if (typeof recomputeSimDuration === 'function') recomputeSimDuration();
      var sidSched = f.standId || null;
      renderFlightList(false, false, { scheduleMode: 'incremental', dirtyFlightIds: [f.id], touchedStandIds: sidSched ? [sidSched] : [] });
    }
    if (arrDepEl) {
      arrDepEl.addEventListener('change', function() {
        if (!state.selectedObject || state.selectedObject.type !== 'flight') return;
        const f = state.selectedObject.obj;
        f.arrDep = this.value === 'Dep' ? 'Dep' : 'Arr';
        if (dwellEl) {
          dwellEl.disabled = f.arrDep !== 'Arr';
          if (f.arrDep !== 'Arr') {
            f.dwellMin = 0;
            dwellEl.value = 0;
          } else {
            f.dwellMin = parseFloat(dwellEl.value) || 0;
          }
        }
        if (minDwellEl) {
          minDwellEl.disabled = f.arrDep !== 'Arr';
          if (f.arrDep !== 'Arr') {
            f.minDwellMin = 0;
            minDwellEl.value = 0;
          } else {
            f.minDwellMin = Math.max(0, parseFloat(minDwellEl.value) || 0);
            minDwellEl.value = f.minDwellMin;
          }
        }
        rebuildSelectedFlightTimeline();
      });
    }
    if (timeInputEl) {
      timeInputEl.addEventListener('change', function() {
        if (!state.selectedObject || state.selectedObject.type !== 'flight') return;
        const f = state.selectedObject.obj;
        const mins = parseTimeToMinutes(this.value || '0');
        f.timeMin = mins;
        this.value = formatMinutesToHHMMSS(mins);
        rebuildSelectedFlightTimeline();
      });
    }
    if (aircraftEl) {
      aircraftEl.addEventListener('change', function() {
        if (!state.selectedObject || state.selectedObject.type !== 'flight') return;
        const f = state.selectedObject.obj;
        f.aircraftType = this.value || 'A320';
        f.code = getCodeForAircraft(f.aircraftType);
        rebuildSelectedFlightTimeline();
      });
    }
    if (regEl) {
      regEl.addEventListener('change', function() {
        if (!state.selectedObject || state.selectedObject.type !== 'flight') return;
        const f = state.selectedObject.obj;
        f.reg = this.value || '';
        if (typeof markGlobalUpdateStale === 'function') markGlobalUpdateStale();
        var rs = f.standId || null;
        renderFlightList(false, false, { scheduleMode: 'incremental', dirtyFlightIds: [f.id], touchedStandIds: rs ? [rs] : [] });
        updateObjectInfo();
      });
    }
    const airlineCodeEl = document.getElementById('flightAirlineCode');
    const flightNumberEl = document.getElementById('flightFlightNumber');
    if (airlineCodeEl) {
      airlineCodeEl.addEventListener('change', function() {
        if (!state.selectedObject || state.selectedObject.type !== 'flight') return;
        const f = state.selectedObject.obj;
        f.airlineCode = this.value || '';
        if (typeof markGlobalUpdateStale === 'function') markGlobalUpdateStale();
        var rs2 = f.standId || null;
        renderFlightList(false, false, { scheduleMode: 'incremental', dirtyFlightIds: [f.id], touchedStandIds: rs2 ? [rs2] : [] });
        updateObjectInfo();
      });
    }
    if (flightNumberEl) {
      flightNumberEl.addEventListener('change', function() {
        if (!state.selectedObject || state.selectedObject.type !== 'flight') return;
        const f = state.selectedObject.obj;
        f.flightNumber = this.value || '';
        if (typeof markGlobalUpdateStale === 'function') markGlobalUpdateStale();
        var rs3 = f.standId || null;
        renderFlightList(false, false, { scheduleMode: 'incremental', dirtyFlightIds: [f.id], touchedStandIds: rs3 ? [rs3] : [] });
        updateObjectInfo();
      });
    }
    if (dwellEl) {
      dwellEl.addEventListener('change', function() {
        if (!state.selectedObject || state.selectedObject.type !== 'flight') return;
        const f = state.selectedObject.obj;
        let v = parseFloat(this.value);
        v = (typeof v === 'number' && !isNaN(v) && v >= 0) ? v : 0;
        let dwell = Math.max(SCHED_DWELL_FLOOR_MIN, v);
        let minDwell = f.minDwellMin != null ? f.minDwellMin : dwell;
        minDwell = Math.max(SCHED_DWELL_FLOOR_MIN, minDwell);
        if (minDwell > dwell) minDwell = dwell;
        f.dwellMin = dwell;
        f.minDwellMin = minDwell;
        this.value = f.dwellMin;
        if (minDwellEl) minDwellEl.value = f.minDwellMin;
        rebuildSelectedFlightTimeline();
      });
    }
    if (minDwellEl) {
      minDwellEl.addEventListener('change', function() {
        if (!state.selectedObject || state.selectedObject.type !== 'flight') return;
        const f = state.selectedObject.obj;
        let dwell = f.dwellMin != null ? f.dwellMin : 0;
        dwell = Math.max(SCHED_DWELL_FLOOR_MIN, dwell);
        let v = parseFloat(this.value);
        v = (typeof v === 'number' && !isNaN(v) && v >= 0) ? v : 0;
        let minDwell = Math.max(SCHED_DWELL_FLOOR_MIN, v);
        if (minDwell > dwell) minDwell = dwell;
        f.dwellMin = dwell;
        f.minDwellMin = minDwell;
        if (dwellEl) dwellEl.value = f.dwellMin;
        this.value = f.minDwellMin;
        if (typeof markGlobalUpdateStale === 'function') markGlobalUpdateStale();
        var rs4 = f.standId || null;
        if (typeof renderFlightList === 'function')
          renderFlightList(false, false, { scheduleMode: 'incremental', dirtyFlightIds: [f.id], touchedStandIds: rs4 ? [rs4] : [] });
      });
    }
    if (playBtn) {
      playBtn.addEventListener('click', function() {
        const errs = validateNetworkForFlights();
        if (errs.length) {
          state.simPlaying = false;
          updateFlightError(errs);
          alert('Simulation cannot be played:\\n' + errs.join('\\n'));
          return;
        }
        if (!state.flights.length) {
          updateFlightError('registered FlightThere is no.');
          alert('registered FlightThere is no.');
          return;
        }
        if (!state.globalUpdateFresh) {
          alert('Update(새로고침)이 필요합니다. 빨간 동기화 표시일 때는 타임라인이 비어 있어 재생할 수 없습니다.');
          return;
        }
        if (typeof recomputeSimDuration === 'function') recomputeSimDuration();
        const lo = state.simStartSec, hi = state.simDurationSec;
        let t = snapSimTimeSecForSlider(Math.max(lo, Math.min(hi, state.simTimeSec)));
        if (hi > lo && t >= hi - 1e-3) t = snapSimTimeSecForSlider(lo);
        state.simTimeSec = t;
        if (simSlider) simSlider.value = state.simTimeSec;
        if (typeof updateFlightSimPlaybackLabelsDom === 'function') updateFlightSimPlaybackLabelsDom();
        if (typeof prepareLazyTimelinesForCurrentSim === 'function') prepareLazyTimelinesForCurrentSim(state.simTimeSec);
        state.simPlaying = true;
        ensureSimLoop._lastTs = null;
        ensureSimLoop._playKick = true;
        ensureSimLoop();
        try { draw(); } catch(e) {}
        if (typeof update3DScene === 'function') update3DScene();
      });
    }
    if (pauseBtn) {
      pauseBtn.addEventListener('click', function() {
        state.simPlaying = false;
        if (typeof ensureSimLoop === 'function') ensureSimLoop._playKick = false;
      });
    }
    if (resetBtn) {
      resetBtn.addEventListener('click', function() {
        state.simPlaying = false;
        if (typeof ensureSimLoop === 'function') ensureSimLoop._playKick = false;
        state.simTimeSec = snapSimTimeSecForSlider(state.simStartSec);
        if (simSlider) simSlider.value = state.simTimeSec;
        if (typeof updateFlightSimPlaybackLabelsDom === 'function') updateFlightSimPlaybackLabelsDom();
        try { draw(); } catch(e) {}
        if (typeof update3DScene === 'function') update3DScene();
      });
    }
    if (simSlider) {
      simSlider.addEventListener('input', function() {
        const secs = parseFloat(this.value);
        if (!isNaN(secs)) {
          const snapped = snapSimTimeSecForSlider(secs);
          state.simTimeSec = snapped;
          this.value = snapped;
          if (typeof updateFlightSimPlaybackLabelsDom === 'function') updateFlightSimPlaybackLabelsDom();
          if (typeof prepareLazyTimelinesForCurrentSim === 'function') prepareLazyTimelinesForCurrentSim(state.simTimeSec);
          try { draw(); } catch(e) {}
          if (typeof update3DScene === 'function') update3DScene();
        }
      });
    }
    if (speedSelect) {
      speedSelect.addEventListener('change', function() {
        const v = parseFloat(this.value);
        state.simSpeed = !isNaN(v) && v > 0 ? v : 1;
      });
      const v0 = parseFloat(speedSelect.value);
      state.simSpeed = !isNaN(v0) && v0 > 0 ? v0 : _dc.defaultSimSpeed;
    }
    const btnHideSimBar = document.getElementById('btnHideSimPlaybackBar');
    if (btnHideSimBar) {
      btnHideSimBar.addEventListener('click', function() {
        state.simPlaybackDockVisible = false;
        if (typeof applySimPlaybackBarDomVisibility === 'function') applySimPlaybackBarDomVisibility();
      });
    }
    function syncTableToFlightState() {
      const schedTable = document.querySelector('.flight-schedule-table');
      if (!schedTable || !Array.isArray(state.flights)) return;
      const rows = Array.from(schedTable.querySelectorAll('tbody tr.flight-data-row'));
      rows.forEach(function(row) {
        const fid = row.getAttribute('data-id');
        if (!fid) return;
        const f = state.flights.find(function(ff) { return ff && ff.id === fid; });
        if (!f) return;
        const tds = Array.from(row.querySelectorAll('td'));
        if (tds.length < 15) return;
        const getMin = function(idx) {
          const txt = (tds[idx] && tds[idx].textContent || '').trim();
          if (!txt) return null;
          const parts = txt.split(':');
          if (parts.length >= 2) {
            const h = parseInt(parts[0], 10) || 0;
            const m = parseInt(parts[1], 10) || 0;
            const s = parts.length >= 3 ? (parseInt(parts[2], 10) || 0) : 0;
            return h * 60 + m + s / 60;
          }
          const n = parseFloat(txt);
          return isNaN(n) ? null : n;
        };
        const map = {
          sldtMin_d: 7, sibtMin_d: 8, sobtMin_d: 9,  stotMin_d: 10,
          eldtMin:  11, eibtMin:  12, eobtMin:  13, etotMin:   14
        };
        Object.keys(map).forEach(function(key) {
          const v = getMin(map[key]);
          if (v != null) f[key] = v;
        });
      });
    }
    function setLayoutMessage(msg, isError) {
      if (!layoutMsgEl) return;
      layoutMsgEl.textContent = msg || '';
      layoutMsgEl.style.color = isError ? '#f97316' : '#9ca3af';
    }
    if (saveLayoutBtn) {
      saveLayoutBtn.addEventListener('click', function() {
        const name = (layoutNameInput && layoutNameInput.value || '').trim();
        if (!name) {
          setLayoutMessage('Please enter a save name.', true);
          return;
        }
        try {
          if (typeof syncStateFromPanel === 'function') syncStateFromPanel();
          if (typeof syncTableToFlightState === 'function') syncTableToFlightState();
          const data = serializeCurrentLayout();
          fetchSaveLayout(name, data).then(function(r) {
            if (r.ok) {
              if (typeof updateLayoutNameBar === 'function') updateLayoutNameBar(name);
              setLayoutMessage('Saved to Layout_storage as "' + name + '.json"', false);
            } else setLayoutMessage('save failed (status ' + r.status + ') — python run_app.pyAfter running with http://127.0.0.1:8501 connection', true);
          }).catch(function(e) {
            console.warn('Layout save fetch failed', e);
            setLayoutMessage('Connection failed: ' + (e && e.message) + ' — python run_app.pyAfter running with http://127.0.0.1:8501 connection', true);
          });
        } catch (e) {
          console.error(e);
          setLayoutMessage('Unable to save layout.', true);
        }
      });
    }
    function switchLayoutTab(tabId) {
      const root = document.getElementById('tab-saveload');
      if (!root) return;
      root.querySelectorAll('.layout-save-load-tab').forEach(btn => btn.classList.remove('active'));
      root.querySelectorAll('.layout-save-load-pane').forEach(p => p.classList.remove('active'));
      const btn = root.querySelector('.layout-save-load-tab[data-sltab="' + tabId + '"]');
      const pane = document.getElementById('layout-' + tabId + '-pane');
      if (btn) btn.classList.add('active');
      if (pane) pane.classList.add('active');
      if (tabId === 'load') fetchAndRefreshLayoutList();
    }
    const layoutMessageSaveEl = document.getElementById('layoutMessageSave');
    const btnSaveCurrent = document.getElementById('btnSaveCurrentLayout');
    if (btnSaveCurrent) btnSaveCurrent.addEventListener('click', function() {
      const name = (state.currentLayoutName && state.currentLayoutName.trim()) || (INITIAL_LAYOUT_DISPLAY_NAME || 'default_layout');
      try {
        if (typeof syncStateFromPanel === 'function') syncStateFromPanel();
        if (typeof syncTableToFlightState === 'function') syncTableToFlightState();
        const data = serializeCurrentLayout();
        fetchSaveLayout(name, data).then(function(r) {
          if (r.ok) {
            if (layoutMessageSaveEl) { layoutMessageSaveEl.textContent = 'saved: ' + name + '.json'; layoutMessageSaveEl.style.color = '#9ca3af'; }
          } else if (layoutMessageSaveEl) { layoutMessageSaveEl.textContent = 'save failed (status ' + r.status + ')'; layoutMessageSaveEl.style.color = '#f97316'; }
        }).catch(function(e) {
          console.warn('Object save fetch failed', e);
          if (layoutMessageSaveEl) { layoutMessageSaveEl.textContent = 'Connection failed: ' + (e && e.message); layoutMessageSaveEl.style.color = '#f97316'; }
        });
      } catch (e) { if (layoutMessageSaveEl) { layoutMessageSaveEl.textContent = 'error: ' + (e && e.message); layoutMessageSaveEl.style.color = '#f97316'; } }
    });
    const saveLoadTabRoot = document.getElementById('tab-saveload');
    if (saveLoadTabRoot) {
      saveLoadTabRoot.querySelectorAll('.layout-save-load-tab[data-sltab]').forEach(btn => {
        btn.addEventListener('click', function() { switchLayoutTab(this.getAttribute('data-sltab')); });
      });
    }
    function getLayoutApiBase() {
      if (LAYOUT_API_URL && LAYOUT_API_URL !== 'null') return LAYOUT_API_URL;
      try { if (window.location && window.location.origin && window.location.origin !== 'null') return window.location.origin; } catch(e) {}
      return '';
    }
    function fetchSaveLayout(name, data) {
      const apiBase = (typeof getLayoutApiBase === 'function') ? getLayoutApiBase() : (LAYOUT_API_URL || '');
      return fetch(apiBase + '/api/save-layout', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ layout: data, name: name })
      });
    }
    function fetchAndRefreshLayoutList() {
      if (!layoutLoadListEl) return;
      layoutLoadListEl.innerHTML = '<div style="font-size:11px;color:#9ca3af;">Loading list...</div>';
      const apiBase = getLayoutApiBase();
      fetch(apiBase + '/api/list-layouts').then(function(r) {
        if (!r.ok) throw new Error('API Connection failed (status ' + r.status + ')');
        return r.json();
      }).then(function(data) {
        const names = (data && data.names) ? data.names : (Array.isArray(LAYOUT_NAMES) ? LAYOUT_NAMES : []);
        refreshLayoutLoadList(names);
      }).catch(function(e) {
        console.warn('Layout list fetch failed', e);
        layoutLoadListEl.innerHTML = '<div style="font-size:11px;color:#f97316;">Connection failed: ' + (e && e.message) + '</div><div style="font-size:10px;color:#9ca3af;margin-top:4px;">python run_app.py After running with http://127.0.0.1:8501 connection</div>';
      });
    }
    function refreshLayoutLoadList(namesFromApi) {
      if (!layoutLoadListEl) return;
      const names = namesFromApi != null ? (Array.isArray(namesFromApi) ? namesFromApi : []) : (Array.isArray(LAYOUT_NAMES) ? LAYOUT_NAMES : []);
      if (!names.length) {
        layoutLoadListEl.innerHTML = '<div style="font-size:11px;color:#9ca3af;">There are no saved layouts.</div>';
        return;
      }
      const reserved = { 'default_layout': true, 'current_layout': true };
      layoutLoadListEl.innerHTML = names.map(function(name) {
        const n = (name || '').replace(/"/g, '&quot;').replace(/</g, '&lt;');
        const showDel = !reserved[(name || '').toLowerCase()];
        const delBtn = showDel ? '<button type="button" class="layout-load-delete" title="Delete" data-name="' + (name || '').replace(/"/g, '&quot;') + '">×</button>' : '';
        return '<div class="layout-load-item" data-name="' + (name || '').replace(/"/g, '&quot;') + '"><span class="layout-load-name">' + n + '</span>' + delBtn + '</div>';
      }).join('');
      layoutLoadListEl.querySelectorAll('.layout-load-item').forEach(function(el) {
        const name = el.getAttribute('data-name');
        el.addEventListener('click', function(ev) {
          if (ev.target && ev.target.classList && ev.target.classList.contains('layout-load-delete')) return;
          if (!name) return;
          var apiBase = getLayoutApiBase();
          if (layoutMsgEl) { layoutMsgEl.textContent = 'Loading...'; layoutMsgEl.style.color = '#9ca3af'; }
          fetch(apiBase + '/api/load-layout?name=' + encodeURIComponent(name)).then(function(r) {
            if (!r.ok) throw new Error('not_found');
            return r.json();
          }).then(function(obj) {
            if (!obj || typeof obj !== 'object') { throw new Error('invalid_response'); }
            try {
              state.hasSimulationResult = false;
              applyLayoutObject(obj);
              resizeCanvas();
              reset2DView();
              syncPanelFromState();
              if (typeof draw === 'function') draw();
              if (typeof update3DScene === 'function') update3DScene();
              if (typeof updateLayoutNameBar === 'function') updateLayoutNameBar(name);
              if (typeof recomputeSimDuration === 'function') recomputeSimDuration();
              if (layoutMsgEl) { layoutMsgEl.textContent = 'Loaded \"' + name + '\"'; layoutMsgEl.style.color = '#9ca3af'; }
            } catch (err) {
              console.error('applyLayoutObject error', err);
              throw err;
            }
          }).catch(function(e) {
            console.warn('Layout load fetch failed', e);
            if (layoutMsgEl) { layoutMsgEl.textContent = 'Failed to load: ' + ((e && e.message) || name || '') + ' — python run_app.pyAfter running with http://127.0.0.1:8501 connection'; layoutMsgEl.style.color = '#f97316'; }
          });
        });
        el.querySelector('.layout-load-delete') && el.querySelector('.layout-load-delete').addEventListener('click', function(ev) {
          ev.stopPropagation();
          const n = this.getAttribute('data-name');
          if (!n) return;
          const apiBase = getLayoutApiBase();
          fetch(apiBase + '/api/delete-layout', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: n })
          }).then(function(r) {
            if (!r.ok) return r.json().then(function(d) { throw new Error(d.error || 'Deletion failed'); });
            return fetch(apiBase + '/api/list-layouts').then(function(r2) { return r2.json(); });
          }).then(function(data) {
            if (data && data.names) refreshLayoutLoadList(data.names);
            if (layoutMsgEl) { layoutMsgEl.textContent = 'deleted.'; layoutMsgEl.style.color = '#9ca3af'; }
          }).catch(function(e) {
            console.warn('Layout delete fetch failed', e);
            if (layoutMsgEl) { layoutMsgEl.textContent = ((e && e.message) || 'Deletion failed') + ' — python run_app.pyAfter running with http://127.0.0.1:8501 connection'; layoutMsgEl.style.color = '#f97316'; }
          });
        });
      });
    }
    fetch((getLayoutApiBase() || '') + '/api/list-layouts').then(function(r) {
      if (r.ok) return;
      var banner = document.getElementById('api-warning-banner');
      if (banner) banner.style.display = 'block';
    }).catch(function(e) {
      console.warn('API health check failed', e);
      var banner = document.getElementById('api-warning-banner');
      if (banner) banner.style.display = 'block';
    });
  })();

  document.getElementById('btnTerminalDraw').addEventListener('click', function() {
    state.selectedObject = null;
    if (state.terminalDrawingId) {
      const t = state.terminals.find(x => x.id === state.terminalDrawingId);
      if (t && !t.closed && t.vertices.length >= 3) {
        t.closed = true;
        if (terminalOverlapsAnyTaxiway(t)) {
          alert('this Apron/Terminalsilver Taxiway Overlaps the center line. Please place it in a different location.');
          state.terminals = state.terminals.filter(term => term.id !== t.id);
        }
      }
      state.terminalDrawingId = null;
      state.layoutPathDrawPointer = null;
      syncPanelFromState();
      draw();
      return;
    }
    const selectedBuildingType = normalizeBuildingType(document.getElementById('buildingType') ? document.getElementById('buildingType').value : BUILDING_TYPE_DEFAULT);
    const nameBase = document.getElementById('terminalName').value.trim() || getDefaultBuildingNameForType(selectedBuildingType);
    const floorsEl = document.getElementById('terminalFloors');
    const f2fEl = document.getElementById('terminalFloorToFloor');
    let floors = floorsEl ? parseInt(floorsEl.value, 10) : 1;
    let f2f = f2fEl ? Number(f2fEl.value) : 4;
    floors = Math.max(1, floors || 1);
    f2f = Math.max(0.5, f2f || 4);
    const totalH = floors * f2f;
    if (findDuplicateLayoutName('terminal', null, nameBase)) {
      alertDuplicateLayoutName();
      return;
    }
    const term = { id: id(), name: nameBase, buildingType: selectedBuildingType, vertices: [], closed: false, floors, floorToFloor: f2f, floorHeight: totalH, departureCapacity: 0, arrivalCapacity: 0 };
    pushUndo();
    state.terminals.push(term);
    state.currentTerminalId = term.id;
    state.terminalDrawingId = term.id;
    syncPanelFromState();
    draw();
    if (typeof markGlobalUpdateStale === 'function') markGlobalUpdateStale();
  });

  document.getElementById('btnTaxiwayDraw').addEventListener('click', function() {
    const hadSelection = !!state.selectedObject;
    state.selectedObject = null;
    if (state.taxiwayDrawingId) {
      const tw = state.taxiways.find(x => x.id === state.taxiwayDrawingId);
      if (tw && tw.vertices.length >= 2) {
        if (taxiwayOverlapsAnyTerminal(tw)) {
          alert('this TaxiwayIs TerminalIt overlaps with . Please draw a different path.');
          pushUndo();
          state.taxiways = state.taxiways.filter(t => t.id !== tw.id);
        }
        state.taxiwayDrawingId = null;
        state.layoutPathDrawPointer = null;
        syncPanelFromState();
        if (typeof redrawLayoutAfterEdit === 'function') redrawLayoutAfterEdit();
        else if (typeof updateAllFlightPaths === 'function') updateAllFlightPaths(); else draw();
      }
      return;
    }
    const layoutMode = settingModeSelect ? settingModeSelect.value : 'taxiway';
    const pathType = pathTypeFromLayoutMode(isPathLayoutMode(layoutMode) ? layoutMode : 'taxiway');
    const nameInputEl = document.getElementById('taxiwayName');
    const defaultPathName = getDefaultPathName(pathType);
    if (hadSelection && nameInputEl) nameInputEl.value = defaultPathName;
    const rawName = nameInputEl ? nameInputEl.value.trim() : '';
    const nameBase = rawName || defaultPathName;
    const inputWidth = Number(document.getElementById('taxiwayWidth').value);
    const baseWidth = pathType === 'runway'
      ? RUNWAY_PATH_DEFAULT_WIDTH
      : (pathType === 'runway_exit' ? RUNWAY_EXIT_DEFAULT_WIDTH : TAXIWAY_DEFAULT_WIDTH);
    const widthVal = clampTaxiwayWidthM(pathType, inputWidth, baseWidth);
    const modeVal = (function() {
      const raw = document.getElementById('taxiwayDirectionMode') ? document.getElementById('taxiwayDirectionMode').value : '';
      if (pathType === 'runway') return (raw === 'counter_clockwise') ? 'counter_clockwise' : 'clockwise';
      return raw || 'both';
    })();
    const maxExitInput = document.getElementById('taxiwayMaxExitVel');
    const minExitInput = document.getElementById('taxiwayMinExitVel');
    const maxExitVelocity = (pathType === 'runway_exit' && maxExitInput)
      ? (function() { const mv = Number(maxExitInput.value); return isFinite(mv) && mv > 0 ? mv : null; })()
      : null;
    const minExitVelocity = (pathType === 'runway_exit' && minExitInput)
      ? (function() {
          const mv = Number(minExitInput.value);
          if (!isFinite(mv) || mv <= 0) return 15;
          if (maxExitVelocity != null && mv > maxExitVelocity) return maxExitVelocity;
          return mv;
        })()
      : undefined;
    const allowedRwDirections = (pathType === 'runway_exit')
      ? getRunwayExitAllowedDirectionsFromPanel()
      : undefined;
    const minArrVelInput = document.getElementById('runwayMinArrVelocity');
    const minArrVelocity = (pathType === 'runway' && minArrVelInput)
      ? (function() {
          const mv = Number(minArrVelInput.value);
          return (isFinite(mv) && mv > 0) ? Math.max(1, Math.min(150, mv)) : 15;
        })()
      : undefined;
    const lineupEl = document.getElementById('runwayLineupDistM');
    const lineupDistM = (pathType === 'runway' && lineupEl)
      ? (function() { const x = Number(lineupEl.value); return (isFinite(x) && x >= 0) ? x : 0; })()
      : undefined;
    const runwayStartDispEl = document.getElementById('runwayStartDisplacedThresholdM');
    const startDisplacedThresholdM = (pathType === 'runway' && runwayStartDispEl)
      ? (function() { const x = Number(runwayStartDispEl.value); return (isFinite(x) && x >= 0) ? x : RUNWAY_START_DISPLACED_THRESHOLD_DEFAULT_M; })()
      : undefined;
    const runwayStartBlastEl = document.getElementById('runwayStartBlastPadM');
    const startBlastPadM = (pathType === 'runway' && runwayStartBlastEl)
      ? (function() { const x = Number(runwayStartBlastEl.value); return (isFinite(x) && x >= 0) ? x : RUNWAY_START_BLAST_PAD_DEFAULT_M; })()
      : undefined;
    const runwayEndDispEl = document.getElementById('runwayEndDisplacedThresholdM');
    const endDisplacedThresholdM = (pathType === 'runway' && runwayEndDispEl)
      ? (function() { const x = Number(runwayEndDispEl.value); return (isFinite(x) && x >= 0) ? x : RUNWAY_END_DISPLACED_THRESHOLD_DEFAULT_M; })()
      : undefined;
    const runwayEndBlastEl = document.getElementById('runwayEndBlastPadM');
    const endBlastPadM = (pathType === 'runway' && runwayEndBlastEl)
      ? (function() { const x = Number(runwayEndBlastEl.value); return (isFinite(x) && x >= 0) ? x : RUNWAY_END_BLAST_PAD_DEFAULT_M; })()
      : undefined;
    const taxiway = { id: id(), name: nameBase, vertices: [], width: widthVal, direction: modeVal, pathType, maxExitVelocity, minExitVelocity, allowedRwDirections, minArrVelocity, lineupDistM, avgMoveVelocity: (function() {
      const el = document.getElementById('taxiwayAvgMoveVelocity');
      const v = el ? Number(el.value) : 10;
      return (typeof v === 'number' && isFinite(v) && v > 0) ? Math.max(1, Math.min(50, v)) : 10;
    })(), startDisplacedThresholdM, startBlastPadM, endDisplacedThresholdM, endBlastPadM };
    if (pathType !== 'runway') delete taxiway.minArrVelocity;
    if (pathType !== 'runway') delete taxiway.lineupDistM;
    if (pathType !== 'runway') delete taxiway.startDisplacedThresholdM;
    if (pathType !== 'runway') delete taxiway.startBlastPadM;
    if (pathType !== 'runway') delete taxiway.endDisplacedThresholdM;
    if (pathType !== 'runway') delete taxiway.endBlastPadM;
    if (pathType !== 'runway_exit') { delete taxiway.maxExitVelocity; delete taxiway.minExitVelocity; delete taxiway.allowedRwDirections; }
    if (findDuplicateLayoutName('taxiway', null, nameBase)) {
      alertDuplicateLayoutName();
      return;
    }
    pushUndo();
    state.taxiways.push(taxiway);
    state.taxiwayDrawingId = taxiway.id;
    syncPanelFromState();
    if (typeof redrawLayoutAfterEdit === 'function') redrawLayoutAfterEdit();
    else if (typeof updateAllFlightPaths === 'function') updateAllFlightPaths(); else draw();
  });
  const btnPbbDrawEl = document.getElementById('btnPbbDraw');
  if (btnPbbDrawEl) btnPbbDrawEl.addEventListener('click', function() {
    toggleLayoutDrawMode('pbbDrawing', 'previewPbb', null);
  });
  const btnRemoteDrawEl = document.getElementById('btnRemoteDraw');
  if (btnRemoteDrawEl) btnRemoteDrawEl.addEventListener('click', function() {
    toggleLayoutDrawMode('remoteDrawing', 'previewRemote', null);
  });
  const btnHoldingPointDrawEl = document.getElementById('btnHoldingPointDraw');
  if (btnHoldingPointDrawEl) btnHoldingPointDrawEl.addEventListener('click', function() {
    toggleLayoutDrawMode('holdingPointDrawing', 'previewHoldingPoint', null);
  });
  const btnApronDrawEl = document.getElementById('btnApronLinkDraw');
  if (btnApronDrawEl) btnApronDrawEl.addEventListener('click', function() {
    toggleLayoutDrawMode('apronLinkDrawing', null, 'apronLinkTemp');
  });

  (function setupRightPanelDragResize() {
    if (!panel || !panelToggle) return;
    const rootStyle = () => getComputedStyle(document.documentElement);
    function readPxVar(name, fallback) {
      const v = parseFloat(rootStyle().getPropertyValue(name));
      return Number.isFinite(v) ? v : fallback;
    }
    function readLenVar(name, fallback) {
      const t = (rootStyle().getPropertyValue(name) || '').trim();
      return t || fallback;
    }
    function parseCssLenToPx(s, vwBase) {
      const str = String(s || '').trim().toLowerCase();
      const n = parseFloat(str);
      if (!Number.isFinite(n)) return vwBase * 0.5;
      if (str.endsWith('vw')) return (n / 100) * vwBase;
      if (str.endsWith('vh')) return (n / 100) * (typeof window !== 'undefined' ? window.innerHeight : 800);
      if (str.endsWith('%')) return (n / 100) * vwBase;
      if (str.endsWith('px')) return n;
      return n;
    }
    function maxPanelPx() {
      const m = readPxVar('--style-right-panel-resize-viewport-margin', 8);
      return Math.max(120, window.innerWidth - m);
    }
    function collapsedPx() { return readPxVar('--style-right-panel-resize-collapsed', 44); }
    function collapseBelowPx() { return readPxVar('--style-right-panel-resize-collapse-below', 96); }
    function minExpandedPx() { return readPxVar('--style-right-panel-resize-min-expanded', 220); }
    let lastExpandedWidthPx = Math.round(parseCssLenToPx(readLenVar('--style-right-panel-width-full', '50vw'), window.innerWidth));
    lastExpandedWidthPx = Math.min(maxPanelPx(), Math.max(minExpandedPx(), lastExpandedWidthPx));
    function syncToolbar(px) {
      document.documentElement.style.setProperty('--layout-toolbar-right', Math.round(px) + 'px');
    }
    function applyCollapsed() {
      panel.classList.add('collapsed');
      panel.style.width = '';
      syncToolbar(collapsedPx());
      panelToggle.textContent = '▶';
    }
    function applyExpandedWidthPx(px) {
      const cap = maxPanelPx();
      let w = Math.min(cap, Math.round(px));
      w = Math.max(minExpandedPx(), w);
      panel.classList.remove('collapsed');
      panel.style.width = w + 'px';
      lastExpandedWidthPx = w;
      syncToolbar(w);
      panelToggle.textContent = '◀';
    }
    function applyDragWidthPx(rawPx) {
      const cap = maxPanelPx();
      const c0 = collapsedPx();
      const below = collapseBelowPx();
      let w = Math.min(cap, Math.max(c0, Math.round(rawPx)));
      if (w < below) {
        panel.classList.add('collapsed');
        panel.style.width = '';
        syncToolbar(c0);
        panelToggle.textContent = '▶';
        return;
      }
      panel.classList.remove('collapsed');
      panel.style.width = w + 'px';
      syncToolbar(w);
      panelToggle.textContent = '◀';
    }
    function finishDragWidthPx(rawPx) {
      const below = collapseBelowPx();
      const cap = maxPanelPx();
      let w = Math.min(cap, Math.max(collapsedPx(), Math.round(rawPx)));
      if (w < below) {
        applyCollapsed();
        return;
      }
      w = Math.min(cap, Math.max(minExpandedPx(), w));
      applyExpandedWidthPx(w);
    }
    applyExpandedWidthPx(lastExpandedWidthPx);
    let dragStartClientX = 0;
    let dragStartWidth = 0;
    let lastMoveClientX = 0;
    let dragMoved = false;
    let resizePointerActive = false;
    let suppressToggleClick = false;
    const CLICK_MAX_MOVE = _interactionConfigNum('clickMaxMovePx', 6);
    function onResizeWindow() {
      if (panel.classList.contains('collapsed')) {
        syncToolbar(collapsedPx());
        return;
      }
      const rw = panel.getBoundingClientRect().width;
      const cap = maxPanelPx();
      if (rw > cap) applyExpandedWidthPx(cap);
      else syncToolbar(rw);
    }
    window.addEventListener('resize', onResizeWindow);
    panelToggle.addEventListener('click', function(ev) {
      if (suppressToggleClick) {
        ev.preventDefault();
        ev.stopImmediatePropagation();
        suppressToggleClick = false;
      }
    }, true);
    panelToggle.addEventListener('pointerdown', function(ev) {
      if (ev.pointerType === 'mouse' && ev.button !== 0) return;
      ev.preventDefault();
      dragMoved = false;
      resizePointerActive = true;
      dragStartClientX = ev.clientX;
      lastMoveClientX = ev.clientX;
      const c0 = collapsedPx();
      dragStartWidth = panel.classList.contains('collapsed') ? c0 : panel.getBoundingClientRect().width;
      panel.classList.add('panel-resize-dragging');
      try { panelToggle.setPointerCapture(ev.pointerId); } catch (e) {}
    });
    panelToggle.addEventListener('pointermove', function(ev) {
      if (!resizePointerActive) return;
      if (Math.abs(ev.clientX - dragStartClientX) > CLICK_MAX_MOVE) dragMoved = true;
      lastMoveClientX = ev.clientX;
      const w = dragStartWidth + (dragStartClientX - ev.clientX);
      applyDragWidthPx(w);
    });
    function endPointerDrag(ev) {
      if (!resizePointerActive) return;
      resizePointerActive = false;
      panel.classList.remove('panel-resize-dragging');
      try { if (ev && ev.pointerId != null) panelToggle.releasePointerCapture(ev.pointerId); } catch (e) {}
      if (!dragMoved) {
        if (panel.classList.contains('collapsed')) {
          applyExpandedWidthPx(lastExpandedWidthPx);
        } else {
          lastExpandedWidthPx = Math.max(minExpandedPx(), Math.min(maxPanelPx(), panel.getBoundingClientRect().width));
          applyCollapsed();
        }
        dragMoved = false;
        return;
      }
      suppressToggleClick = true;
      const endX = ev && Number.isFinite(ev.clientX) ? ev.clientX : lastMoveClientX;
      const w = dragStartWidth + (dragStartClientX - endX);
      finishDragWidthPx(w);
      dragMoved = false;
    }
    panelToggle.addEventListener('pointerup', endPointerDrag);
    panelToggle.addEventListener('pointercancel', endPointerDrag);
    panelToggle.addEventListener('lostpointercapture', function(ev) {
      if (resizePointerActive) endPointerDrag(ev);
    });
  })();

  function renderObjectList() {
    if (!objectListEl) return;
    const mode = settingModeSelect.value;
    const seen = {};
    const nameCount = {};
    function uniqueTitle(baseName) {
      nameCount[baseName] = (nameCount[baseName] || 0) + 1;
      return nameCount[baseName] > 1 ? baseName + ' (' + nameCount[baseName] + ')' : baseName;
    }
    const items = [];
    if (mode === 'terminal') {
      state.terminals.forEach((t, idx) => {
        if (seen['terminal_' + t.id]) return;
        seen['terminal_' + t.id] = true;
        const areaM2 = t.vertices && t.vertices.length >= 3 ? polygonAreaM2(t.vertices) : 0;
        const floors = t.floors != null ? Math.max(1, parseInt(t.floors, 10) || 1) : 1;
        const f2fRaw = t.floorToFloor != null ? Number(t.floorToFloor) : (t.floorHeight != null ? Number(t.floorHeight) : 4);
        const f2f = Math.max(0.5, f2fRaw || 4);
        const floorH = t.floorHeight != null ? Number(t.floorHeight) || (floors * f2f) : (floors * f2f);
        const dep = t.departureCapacity != null ? t.departureCapacity : 0;
        const arr = t.arrivalCapacity != null ? t.arrivalCapacity : 0;
        const baseName = (t.name && t.name.trim()) ? t.name.trim() : ('Building ' + (idx + 1));
        const buildingTheme = getBuildingTheme(t);
        items.push({
          type: 'terminal',
          id: t.id,
          title: uniqueTitle('Building | ' + baseName),
          tag: 'Height ' + floorH.toFixed(1) + ' m',
          details:
            'Type: ' + buildingTheme.label +
            '<br>' +
            'Area: ' + areaM2.toFixed(1) + ' m²' +
            '<br>Height: ' + floorH.toFixed(1) + ' m' +
            '<br>Floors: ' + floors +
            '<br>Total floor area: ' + (areaM2 * floors).toFixed(1) + ' m²' +
            '<br>Departure: ' + dep +
            '<br>Arrival: ' + arr
        });
      });
    } else if (mode === 'pbb') {
      state.pbbStands.forEach((pbb, idx) => {
        if (seen['pbb_' + pbb.id]) return;
        seen['pbb_' + pbb.id] = true;
        const baseName = (pbb.name && pbb.name.trim()) ? pbb.name.trim() : ('Contact Stand ' + (idx + 1));
        items.push({
          type: 'pbb',
          id: pbb.id,
          title: uniqueTitle('Contact Stand | ' + baseName),
          tag: 'Category ' + (pbb.category || 'C'),
          details: 'Edge cell: (' + pbb.edgeCol + ',' + pbb.edgeRow + ')'
        });
      });
    } else if (mode === 'remote') {
      state.remoteStands.forEach((st, idx) => {
        if (seen['remote_' + st.id]) return;
        seen['remote_' + st.id] = true;
        const baseName = (st.name && st.name.trim()) ? st.name.trim() : ('R' + String(idx + 1).padStart(3, '0'));
        let allowedLabel = 'All (by proximity)';
        if (Array.isArray(st.allowedTerminals) && st.allowedTerminals.length) {
          const terms = makeUniqueNamedCopy(state.terminals || [], 'name').map(function(t) { return {
            id: t.id,
            name: (t.name || '').trim() || 'Building'
          }; });
          const names = st.allowedTerminals.map(function(id) {
            const tt = terms.find(function(t) { return t.id === id; });
            return tt ? tt.name : id;
          });
          if (names.length) allowedLabel = names.join(', ');
        }
        const [rcx, rcy] = getRemoteStandCenterPx(st);
        const rcol = rcx / CELL_SIZE;
        const rrow = rcy / CELL_SIZE;
        items.push({
          type: 'remote',
          id: st.id,
          title: uniqueTitle('Remote stand | ' + baseName),
          tag: 'Category ' + (st.category || 'C'),
          details:
            'Category: ' + (st.category || '—') +
            '<br>Position: (' + rcol.toFixed(1) + ',' + rrow.toFixed(1) + ')' +
            '<br>Angle: ' + normalizeAngleDeg(st.angleDeg != null ? st.angleDeg : 0).toFixed(0) + '°' +
            '<br>available buildings: ' + allowedLabel
        });
      });
    } else if (isPathLayoutMode(mode)) {
      const wantPt = pathTypeFromLayoutMode(mode);
      state.taxiways.forEach((tw, idx) => {
        if (seen['taxiway_' + tw.id]) return;
        const pt = tw.pathType || 'taxiway';
        if (pt !== wantPt) return;
        seen['taxiway_' + tw.id] = true;
        const baseName = (tw.name && tw.name.trim()) ? tw.name.trim() : ('Taxiway ' + (idx + 1));
        const dirVal = getTaxiwayDirection(tw);
        const dirLabel = dirVal === 'clockwise' ? 'CW' : (dirVal === 'counter_clockwise' ? 'CCW' : 'Both');
        let lengthM = 0;
        if (tw.vertices && tw.vertices.length >= 2) {
          for (let i = 1; i < tw.vertices.length; i++) {
            const v0 = tw.vertices[i - 1];
            const v1 = tw.vertices[i];
            const dx = v1.col - v0.col;
            const dy = v1.row - v0.row;
            lengthM += CELL_SIZE * Math.hypot(dx, dy);
          }
        }
        const widthDefault = tw.pathType === 'runway'
          ? RUNWAY_PATH_DEFAULT_WIDTH
          : (tw.pathType === 'runway_exit' ? RUNWAY_EXIT_DEFAULT_WIDTH : TAXIWAY_DEFAULT_WIDTH);
        const widthVal = tw.width != null ? tw.width : widthDefault;
        const serTw = serializeTaxiwayWithEndpoints(tw);
        const startStr = serTw.start_point != null ? '(' + serTw.start_point.col + ',' + serTw.start_point.row + ')' : '—';
        const endStr = serTw.end_point != null ? '(' + serTw.end_point.col + ',' + serTw.end_point.row + ')' : '—';
        const heading = tw.pathType === 'runway' ? 'Runway' : (tw.pathType === 'runway_exit' ? 'Runway Taxiway' : 'Taxiway');
        const avgVel = (typeof tw.avgMoveVelocity === 'number' && isFinite(tw.avgMoveVelocity) && tw.avgMoveVelocity > 0) ? tw.avgMoveVelocity : 10;
        const maxExit = (tw.pathType === 'runway_exit' && typeof tw.maxExitVelocity === 'number' && isFinite(tw.maxExitVelocity) && tw.maxExitVelocity > 0) ? tw.maxExitVelocity : null;
        const minExit = (tw.pathType === 'runway_exit' && typeof tw.minExitVelocity === 'number' && isFinite(tw.minExitVelocity) && tw.minExitVelocity > 0)
          ? (maxExit != null && tw.minExitVelocity > maxExit ? maxExit : tw.minExitVelocity)
          : null;
        const minArrDisplay = tw.pathType === 'runway'
          ? ((typeof tw.minArrVelocity === 'number' && isFinite(tw.minArrVelocity) && tw.minArrVelocity > 0)
            ? Math.max(1, Math.min(150, tw.minArrVelocity))
            : 15)
          : null;
        items.push({
          type: 'taxiway',
          id: tw.id,
          title: uniqueTitle(heading + ' | ' + baseName),
          tag: dirLabel,
          details:
            'Length: ' + lengthM.toFixed(0) + ' m' +
            '<br>Points: ' + tw.vertices.length +
            '<br>Width: ' + widthVal + ' m' +
            (maxExit != null ? '<br>Max exit velocity: ' + maxExit + ' m/s' : '') +
            (minExit != null ? '<br>Min exit velocity: ' + minExit + ' m/s' : '') +
            (minArrDisplay != null ? '<br>Min arr velocity: ' + minArrDisplay + ' m/s' : '') +
            (tw.pathType === 'runway' ? '<br>Line up: ' + getEffectiveRunwayLineupDistM(tw) + ' m (start→end)' : '') +
            (tw.pathType === 'taxiway' ? '<br>Avg move velocity: ' + avgVel + ' m/s' : '') +
            '<br>Start point: ' + startStr +
            '<br>End point: ' + endStr
        });
      });
    } else if (mode === 'holdingPoint') {
      (state.holdingPoints || []).forEach(function(hp, idx) {
        if (!hp || seen['hp_' + hp.id]) return;
        seen['hp_' + hp.id] = true;
        const kindLabel = holdingPointKindDisplayLabel(hp.hpKind);
        const baseName = (hp.name && hp.name.trim()) ? hp.name.trim() : (kindLabel + ' ' + (idx + 1));
        const cx = Number(hp.x), cy = Number(hp.y);
        const col = cx / CELL_SIZE, row = cy / CELL_SIZE;
        const tagShort = normalizeHoldingPointKind(hp.hpKind) === 'runway_holding' ? 'RHP' : 'IHP';
        items.push({
          type: 'holdingPoint',
          id: hp.id,
          title: uniqueTitle(kindLabel + ' | ' + baseName),
          tag: tagShort + ' · ' + c2dHoldingPointDiameterM().toFixed(0) + ' m',
          details:
            'Type: ' + kindLabel +
            '<br>Position (cell): (' + col.toFixed(1) + ', ' + row.toFixed(1) + ')' +
            '<br>World: (' + cx.toFixed(0) + ', ' + cy.toFixed(0) + ')'
        });
      });
    } else if (mode === 'apronTaxiway') {
      state.apronLinks.forEach((lk, idx) => {
        if (seen['apron_' + lk.id]) return;
        seen['apron_' + lk.id] = true;
        const stand = findStandById(lk.pbbId);
        const tw = state.taxiways.find(t => t.id === lk.taxiwayId);
        const title = getApronLinkDisplayName(lk);
        const standLabel = stand && stand.name ? stand.name : lk.pbbId;
        const details = 'Stand: ' + standLabel +
          ', Taxiway: ' + (tw && tw.name ? tw.name : lk.taxiwayId);
        items.push({
          type: 'apronLink',
          id: lk.id,
          title: uniqueTitle('Apron–Taxiway | ' + title),
          tag: 'Apron–Taxiway',
          details
        });
      });
    } else if (mode === 'edge') {
      rebuildDerivedGraphEdges();
      (state.derivedGraphEdges || []).forEach(function(ed) {
        items.push({
          type: 'layoutEdge',
          id: ed.id,
          title: 'Edge | ' + getLayoutEdgeDisplayName(ed),
          tag: 'Graph',
          details:
            'Length (graph): ' + Math.round(ed.dist) +
            '<br>Pixel span: (' + ed.x1.toFixed(0) + ', ' + ed.y1.toFixed(0) + ') → (' + ed.x2.toFixed(0) + ', ' + ed.y2.toFixed(0) + ')' +
            '<br>Polyline points: ' + ((ed.pts && ed.pts.length) ? ed.pts.length : 2) +
            '<br>Node indices: ' + ed.fromIdx + ' → ' + ed.toIdx,
          noDelete: true
        });
      });
    }
    if (!items.length) {
      objectListEl.innerHTML = '<div class="obj-item">No objects yet.</div>';
      return;
    }
    objectListEl.innerHTML = items.map(it => (
      '<div class="obj-item" data-type="' + it.type + '" data-id="' + it.id + '">' +
        '<div class="obj-item-header">' +
          '<span class="obj-item-title">' + it.title + '</span>' +
          '<span class="obj-item-tag">' + it.tag + '</span>' +
          '<button type="button" class="obj-item-delete" title="Delete"' + (it.noDelete ? ' style="display:none" tabindex="-1" aria-hidden="true"' : '') + '>×</button>' +
        '</div>' +
        '<div class="obj-item-details">' + it.details + '</div>' +
      '</div>'
    )).join('');
    const listItems = objectListEl.querySelectorAll('.obj-item');
    listItems.forEach(el => {
      const type = el.getAttribute('data-type');
      const id = el.getAttribute('data-id');
      el.querySelector('.obj-item-delete').addEventListener('click', function(ev) {
        ev.stopPropagation();
        pushUndo();
        removeLayoutObjectFromState(type, id);
        if (state.selectedObject && state.selectedObject.type === type && state.selectedObject.id === id)
          state.selectedObject = null;
        if (type === 'terminal' && state.currentTerminalId === id) {
          state.currentTerminalId = state.terminals.length ? state.terminals[0].id : null;
          if (state.terminalDrawingId === id) {
            state.terminalDrawingId = null;
            state.layoutPathDrawPointer = null;
          }
        }
        if (type === 'taxiway' && state.taxiwayDrawingId === id) {
          state.taxiwayDrawingId = null;
          state.layoutPathDrawPointer = null;
        }
        syncPanelFromState();
        updateObjectInfo();
        if (typeof redrawLayoutAfterEdit === 'function') redrawLayoutAfterEdit();
        else if (typeof updateAllFlightPaths === 'function') updateAllFlightPaths(); else draw();
      });
      el.addEventListener('click', function(ev) {
        if (ev.target.classList.contains('obj-item-delete')) return;
        const typ = this.getAttribute('data-type');
        const idr = this.getAttribute('data-id');
        if (typ === 'layoutEdge') rebuildDerivedGraphEdges();
        const obj = findLayoutObjectByListType(typ, idr);
        if (!obj) return;
        const wasExpanded = this.classList.contains('expanded');
        listItems.forEach(li => li.classList.remove('selected', 'expanded'));
        if (!wasExpanded) {
          this.classList.add('selected', 'expanded');
          state.flightPathRevealFlightId = null;
          state.selectedObject = { type: typ, id: idr, obj };
          if (typ === 'terminal') state.currentTerminalId = idr;
          syncPanelFromState();
          updateObjectInfo();
        } else {
          objectInfoEl.textContent = 'Select an object on the grid or from the list.';
        }
        draw();
      });
    });
    if (state.selectedObject) {
      const sel = objectListEl.querySelector('.obj-item[data-type="' + state.selectedObject.type + '"][data-id="' + state.selectedObject.id + '"]');
      if (sel) sel.classList.add('selected', 'expanded');
    }
  }

  function updateObjectInfo() {
    if (state.selectedObject) {
      const o = state.selectedObject.obj;
      if (state.selectedObject.type === 'terminal') {
        const areaM2 = o.vertices && o.vertices.length >= 3 ? polygonAreaM2(o.vertices) : 0;
        const floors = o.floors != null ? Math.max(1, parseInt(o.floors, 10) || 1) : 1;
        const f2fRaw = o.floorToFloor != null ? Number(o.floorToFloor) : (o.floorHeight != null ? Number(o.floorHeight) : 4);
