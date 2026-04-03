                  .then(function(r) {
                    if (!r.ok) throw new Error('시뮬 결과를 불러오지 못했습니다.');
                    return r.json();
                  })
                  .then(function(data) {
                    if (typeof applyAirsideSimulationResultPayload === 'function') applyAirsideSimulationResultPayload(data);
                  })
                  .catch(function(e) {
                    console.warn('Pro Sim result fetch', e && e.message ? e.message : e);
                  })
                  .finally(function() {
                    if (applyBtnEl) applyBtnEl.disabled = false;
                  });
              })
              .catch(function(e) {
                failProSim(e && e.message ? e.message : 'sim-progress 요청 실패');
              });
          }
          pollProgress();
        }).catch(function(e) {
          failProSim(e && e.message ? e.message : String(e));
        });
      });
    }
    const btnApplySimResult = document.getElementById('btnApplySimResult');
    if (btnApplySimResult) {
      btnApplySimResult.addEventListener('click', function() {
        const base = proSimApiBase();
        if (!base) {
          alert('Layout API가 설정되지 않았습니다.');
          return;
        }
        const layoutName = (state.currentLayoutName && String(state.currentLayoutName).trim()) || INITIAL_LAYOUT_DISPLAY_NAME || 'default_layout';
        btnApplySimResult.disabled = true;
        fetch(base + '/api/load-sim-result?name=' + encodeURIComponent(layoutName))
          .then(function(r) {
            if (!r.ok) throw new Error('시뮬 결과를 찾을 수 없습니다. Pro Sim을 먼저 완료하세요.');
            return r.json();
          })
          .then(function(data) {
            applyAirsideSimulationResultPayload(data);
          })
          .catch(function(e) {
            console.error('Apply sim result', e);
            alert(e && e.message ? e.message : 'Apply failed');
            btnApplySimResult.disabled = false;
          });
      });
    }
    const btnShowPlayDock = document.getElementById('btnShowPlayDock');
    if (btnShowPlayDock) {
      btnShowPlayDock.addEventListener('click', function() {
        state.simPlaybackDockVisible = true;
        if (typeof applySimPlaybackBarDomVisibility === 'function') applySimPlaybackBarDomVisibility();
      });
    }
    function applyTokenNodesFromCheckboxes() {
      const nodes = [];
      TOKEN_NODE_ORDER.forEach((node, i) => {
        const cb = document.getElementById('token' + node.charAt(0).toUpperCase() + node.slice(1));
        if (cb && cb.checked) nodes.push(node);
        else return;
      });
      return nodes;
    }
    function setTokenCheckboxesFromNodes(nodes) {
      const arr = Array.isArray(nodes) ? nodes : [];
      TOKEN_NODE_ORDER.forEach((node, i) => {
        const cb = document.getElementById('token' + node.charAt(0).toUpperCase() + node.slice(1));
        if (cb) cb.checked = arr.indexOf(node) >= 0;
      });
      updateTokenPanesVisibility(arr.length ? arr : TOKEN_NODE_ORDER);
    }
    ['Runway','Taxiway','Apron','Building'].forEach((name, i) => {
      const cb = document.getElementById('token' + name);
      if (!cb) return;
      cb.addEventListener('change', function() {
        if (!state.selectedObject || state.selectedObject.type !== 'flight') return;
        const f = state.selectedObject.obj;
        if (!f.token) f.token = { nodes: TOKEN_NODE_ORDER.slice(), runwayId: null, apronId: null, terminalId: null };
        if (this.checked) {
          f.token.nodes = TOKEN_NODE_ORDER.slice(0, i + 1);
          setTokenCheckboxesFromNodes(f.token.nodes);
        } else {
          f.token.nodes = TOKEN_NODE_ORDER.slice(0, i);
          setTokenCheckboxesFromNodes(f.token.nodes);
        }
        updateTokenPanesVisibility(f.token.nodes);
        rebuildSelectedFlightTimeline();
      });
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
        f.deferPathCompute = true;
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
      if (isArr && arrivalAirsideBlocked(f)) {
        updateFlightError(f.arrRetFailed && !f.noWayArr ? '도착 경로(RET)를 계산하지 못했습니다.' : '도착 경로를 찾을 수 없습니다.');
        f.timeline = null;
        draw();
        return;
      }
      if (!isArr && f.noWayDep) {
        updateFlightError('출발 경로를 찾을 수 없습니다.');
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
          alert('Pro Sim(새로고침)이 필요합니다. 빨간 동기화 표시일 때는 타임라인이 비어 있어 재생할 수 없습니다.');
          return;
        }
        if (typeof recomputeSimDuration === 'function') recomputeSimDuration();
        const lo = state.simStartSec, hi = state.simDurationSec;
        let t = snapSimTimeSecForSlider(Math.max(lo, Math.min(hi, state.simTimeSec)));
        if (hi > lo && t >= hi - 1e-3) t = snapSimTimeSecForSlider(lo);
        state.simTimeSec = t;
        if (simSlider) simSlider.value = state.simTimeSec;
        state.simSliderScrubbing = false;
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
    let simSliderPointerActive = false;
    function finalizeSimSliderPointerDrag() {
      if (!simSliderPointerActive) return;
      simSliderPointerActive = false;
      state.simSliderScrubbing = false;
      if (typeof prepareLazyTimelinesForCurrentSim === 'function') prepareLazyTimelinesForCurrentSim(state.simTimeSec);
      if (typeof updateFlightSimPlaybackLabelsDom === 'function') updateFlightSimPlaybackLabelsDom();
      try { draw(); } catch(e) {}
      if (typeof update3DScene === 'function') update3DScene();
    }
    if (simSlider) {
      simSlider.addEventListener('pointerdown', function(e) {
        if (e.button != null && e.button !== 0) return;
        if (e.isPrimary === false) return;
        simSliderPointerActive = true;
        state.simSliderScrubbing = true;
        try { simSlider.setPointerCapture(e.pointerId); } catch (err) {}
      });
      simSlider.addEventListener('pointerup', function(e) {
        if (!simSliderPointerActive) return;
        try { simSlider.releasePointerCapture(e.pointerId); } catch (err2) {}
        finalizeSimSliderPointerDrag();
      });
      simSlider.addEventListener('pointercancel', function() {
        finalizeSimSliderPointerDrag();
      });
      simSlider.addEventListener('lostpointercapture', function() {
        finalizeSimSliderPointerDrag();
      });
      simSlider.addEventListener('input', function() {
        const secs = parseFloat(this.value);
        if (!isNaN(secs)) {
          const snapped = snapSimTimeSecForSlider(secs);
          state.simTimeSec = snapped;
          this.value = snapped;
          if (typeof updateFlightSimPlaybackLabelsDom === 'function') updateFlightSimPlaybackLabelsDom();
          if (state.simSliderScrubbing) return;
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
    function uniqueTitle(baseName) {
      return baseName;
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
        const f2f = Math.max(0.5, f2fRaw || 4);
        const floorH = o.floorHeight != null ? Number(o.floorHeight) || (floors * f2f) : (floors * f2f);
        const totalArea = areaM2 * floors;
        const dep = o.departureCapacity != null ? o.departureCapacity : 0;
        const arr = o.arrivalCapacity != null ? o.arrivalCapacity : 0;
        objectInfoEl.innerHTML = '<strong>Building</strong><br>Name: ' + (o.name || o.id) + '<br>Type: ' + getBuildingTypeLabel(o.buildingType) + '<br>Vertices: ' + (o.vertices ? o.vertices.length : 0) +
          '<br>Footprint area: ' + areaM2.toFixed(1) + ' m²<br>Height: ' + floorH.toFixed(1) + ' m (Floors: ' + floors + ' × ' + f2f.toFixed(1) + ' m)' +
          '<br>Total floor area: ' + totalArea.toFixed(1) + ' m²' +
          '<br>Departure capacity: ' + dep + '<br>Arrival capacity: ' + arr;
      } else if (state.selectedObject.type === 'pbb') {
        objectInfoEl.innerHTML = '<strong>Contact Stand</strong><br>Name: ' + (o.name || '—') + '<br>Constraint: ' + (getStandCategoryMode(o) === 'aircraft' ? 'Aircraft Type' : ('ICAO ' + (o.category || '—'))) + '<br>PBB count: ' + Math.max(1, parseInt(o.pbbCount, 10) || 1) + '<br>Edge cell: (' + o.edgeCol + ',' + o.edgeRow + ')';
      } else if (state.selectedObject.type === 'remote') {
        let allowedLabel = 'All (by proximity)';
        if (Array.isArray(o.allowedTerminals) && o.allowedTerminals.length) {
          const terms = makeUniqueNamedCopy(state.terminals || [], 'name').map(function(t) { return {
            id: t.id,
            name: (t.name || '').trim() || 'Building'
          }; });
          const names = o.allowedTerminals.map(function(id) {
            const tt = terms.find(function(t) { return t.id === id; });
            return tt ? tt.name : id;
          });
          if (names.length) allowedLabel = names.join(', ');
        }
        const remotePx = getRemoteStandCenterPx(o);
        const remoteCell = [remotePx[0] / CELL_SIZE, remotePx[1] / CELL_SIZE];
        objectInfoEl.innerHTML =
          '<strong>Remote stand</strong>' +
          '<br>Name: ' + (o.name || '—') +
          '<br>Constraint: ' + (getStandCategoryMode(o) === 'aircraft' ? 'Aircraft Type' : ('ICAO ' + (o.category || '—'))) +
          '<br>Cell: (' + remoteCell[0].toFixed(1) + ',' + remoteCell[1].toFixed(1) + ')' +
          '<br>available buildings: ' + allowedLabel;
      } else if (state.selectedObject.type === 'holdingPoint') {
        const hx = Number(o.x), hy = Number(o.y);
        const hCol = hx / CELL_SIZE, hRow = hy / CELL_SIZE;
        objectInfoEl.innerHTML =
          '<strong>' + holdingPointKindDisplayLabel(o.hpKind) + '</strong>' +
          '<br>Name: ' + (o.name || '—') +
          '<br>Diameter: ' + c2dHoldingPointDiameterM().toFixed(0) + ' m' +
          '<br>Cell: (' + hCol.toFixed(1) + ', ' + hRow.toFixed(1) + ')' +
          '<br>World: (' + hx.toFixed(0) + ', ' + hy.toFixed(0) + ')';
      }
      else if (state.selectedObject.type === 'taxiway') {
        const dirVal = getTaxiwayDirection(o);
        const dirLabel = dirVal === 'clockwise' ? 'Clockwise' : (dirVal === 'counter_clockwise' ? 'Counter Clockwise' : 'Both');
        const heading = o.pathType === 'runway' ? 'Runway' : (o.pathType === 'runway_exit' ? 'Runway Taxiway' : 'Taxiway');
        const ser = serializeTaxiwayWithEndpoints(o);
        const startStr = ser.start_point != null ? '(' + ser.start_point.col + ', ' + ser.start_point.row + ')' : '—';
        const endStr = ser.end_point != null ? '(' + ser.end_point.col + ', ' + ser.end_point.row + ')' : '—';
        const avgVel = (typeof o.avgMoveVelocity === 'number' && isFinite(o.avgMoveVelocity) && o.avgMoveVelocity > 0) ? o.avgMoveVelocity : 10;
        const minArr = (o.pathType === 'runway')
          ? ((typeof o.minArrVelocity === 'number' && isFinite(o.minArrVelocity) && o.minArrVelocity > 0) ? Math.max(1, Math.min(150, o.minArrVelocity)) : 15)
          : null;
        const lineupStr = (o.pathType === 'runway') ? (String(getEffectiveRunwayLineupDistM(o)) + ' m (from start toward end)') : '';
        const maxEx = (o.pathType === 'runway_exit' && typeof o.maxExitVelocity === 'number' && isFinite(o.maxExitVelocity) && o.maxExitVelocity > 0) ? o.maxExitVelocity : null;
        const minEx = (o.pathType === 'runway_exit' && typeof o.minExitVelocity === 'number' && isFinite(o.minExitVelocity) && o.minExitVelocity > 0) ? o.minExitVelocity : null;
        objectInfoEl.innerHTML = '<strong>' + heading + '</strong><br>Name: ' + (o.name || '—') +
          '<br>Direction: ' + dirLabel +
          '<br>Width: ' + (o.width != null ? o.width : 23) + ' m' +
          (o.pathType === 'taxiway' ? '<br>Avg move velocity: ' + avgVel + ' m/s' : '') +
          (minArr != null ? '<br>Min arr velocity: ' + minArr + ' m/s' : '') +
          (o.pathType === 'runway' ? '<br>Line up: ' + lineupStr : '') +
          (maxEx != null ? '<br>Max exit velocity: ' + maxEx + ' m/s' : '') +
          (minEx != null ? '<br>Min exit velocity: ' + minEx + ' m/s' : '') +
          '<br>Points: ' + (o.vertices ? o.vertices.length : 0) +
          '<br>Start point: ' + startStr + '<br>End point: ' + endStr;
      } else if (state.selectedObject.type === 'apronLink') {
        const lk = o;
        const stand = findStandById(lk.pbbId);
        const tw = state.taxiways.find(function(t) { return t.id === lk.taxiwayId; });
        objectInfoEl.innerHTML =
          '<strong>Apron Taxiway</strong><br>' +
          'Name: ' + getApronLinkDisplayName(lk) +
          '<br>Stand: ' + (stand && stand.name ? stand.name : lk.pbbId) +
          '<br>Taxiway: ' + (tw && tw.name ? tw.name : lk.taxiwayId) +
          '<br>Link point: (' + Number(lk.tx).toFixed(0) + ', ' + Number(lk.ty).toFixed(0) + ')';
      } else if (state.selectedObject.type === 'layoutEdge') {
        const ed = state.selectedObject.obj;
        objectInfoEl.innerHTML =
          '<strong>Edge (derived)</strong><br>' +
          'Name: ' + getLayoutEdgeDisplayName(ed) +
          '<br>Graph length: ' + (ed && ed.dist != null ? Math.round(ed.dist) : '—') +
          '<br>Nodes: ' + (ed ? ed.fromIdx + ' → ' + ed.toIdx : '—') +
          '<br>Span (px): (' + (ed ? ed.x1.toFixed(0) : '—') + ', ' + (ed ? ed.y1.toFixed(0) : '—') + ') → (' + (ed ? ed.x2.toFixed(0) : '—') + ', ' + (ed ? ed.y2.toFixed(0) : '—') + ')' +
          '<br>Polyline points: ' + (ed && ed.pts ? ed.pts.length : 2);
      } else if (state.selectedObject.type === 'flight') {
        const dir = o.arrDep === 'Dep' ? 'Departure' : 'Arrival';
        const sibt = formatMinutesToHHMMSS(o.sibtMin_d != null ? o.sibtMin_d : (o.timeMin != null ? o.timeMin : 0));
        const sobt = formatMinutesToHHMMSS(o.sobtMin_d != null ? o.sobtMin_d : ((o.timeMin != null ? o.timeMin : 0) + (o.dwellMin != null ? o.dwellMin : 0)));
        const ac = typeof getAircraftInfoByType === 'function' ? getAircraftInfoByType(o.aircraftType) : null;
        const acName = ac ? (ac.name || ac.id || '') : (o.aircraftType || '—');
        const codeIcao = (ac && ac.icao) ? ac.icao : (o.code || '—');
        const icaoJhl = (ac && ac.icaoJHL) ? ac.icaoJHL : '—';
        const recatEu = (ac && ac.recatEu) ? ac.recatEu : '—';
        objectInfoEl.innerHTML =
          '<strong>Flight</strong><br>' +
          'Type: ' + dir +
          '<br>SIBT: ' + sibt + ' &nbsp; SOBT: ' + sobt +
          '<br>Aircraft: ' + (acName || '—') +
          '<br>Code(ICAO): ' + (codeIcao || '—') + ' &nbsp; ICAO(J/H/M/L): ' + (icaoJhl || '—') + ' &nbsp; RECAT-EU: ' + (recatEu || '—') +
          '<br>Reg: ' + (o.reg || '—') +
          '<br>Airline Code: ' + (o.airlineCode || '—') + ' &nbsp; Flight Number: ' + (o.flightNumber || '—') +
          '<br>Dwell (Arr only): ' + (o.dwellMin || 0) + ' min';
      }
    } else
      objectInfoEl.textContent = 'Select an object on the grid or from the list.';
    renderObjectList();
  }

  function reset2DView() {
    let w = 0, h = 0;
    const rect = container.getBoundingClientRect();
    w = Number(rect.width) || 0;
    h = Number(rect.height) || 0;
    if (w <= 0 || h <= 0) {
      if (canvas) {
        w = canvas.clientWidth || canvas.width || 800;
        h = canvas.clientHeight || canvas.height || 600;
      } else {
        w = 800;
        h = 600;
      }
    }
    w = Math.max(1, w);
    h = Math.max(1, h);
    const maxX = GRID_COLS * CELL_SIZE;
    const maxY = GRID_ROWS * CELL_SIZE;
    const scaleX = w / maxX;
    const scaleY = h / maxY;
    const s = Math.min(scaleX, scaleY) * 0.9;
    state.scale = s;
    state.panX = (w - maxX * s) / 2;
    state.panY = (h - maxY * s) / 2;
    draw();
  }

  function resizeCanvas() {
    if (!container || !canvas || !ctx) return;
    const rect = container.getBoundingClientRect();
    const w = Math.max(1, Number(rect.width) || 0);
    const h = Math.max(1, Number(rect.height) || 0);
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    invalidateGridUnderlay();
    safeDraw();
  }

  let _gridUnderlayCanvas = null;
  let _gridUnderlayDirty = true;
  function invalidateGridUnderlay() { _gridUnderlayDirty = true; }
  function rebuildGridUnderlay() {
    const maxX = GRID_COLS * CELL_SIZE, maxY = GRID_ROWS * CELL_SIZE;
    if (!_gridUnderlayCanvas) _gridUnderlayCanvas = document.createElement('canvas');
    _gridUnderlayCanvas.width = Math.max(1, Math.floor(maxX * dpr));
    _gridUnderlayCanvas.height = Math.max(1, Math.floor(maxY * dpr));
    const uctx = _gridUnderlayCanvas.getContext('2d');
    uctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    uctx.fillStyle = GRID_VIEW_BG;
    uctx.fillRect(0, 0, maxX, maxY);
    if (state.layoutImageOverlay && layoutImageBitmap) {
      const overlay = state.layoutImageOverlay;
      const [imgX, imgY] = cellToPixel(overlay.topLeftCol, overlay.topLeftRow);
      uctx.save();
      uctx.globalAlpha = state.showImage ? clampLayoutImageOpacity(overlay.opacity) : 0;
      uctx.imageSmoothingEnabled = true;
      uctx.drawImage(
        layoutImageBitmap,
        imgX,
        imgY,
        clampLayoutImageSize(overlay.widthM, GRID_LAYOUT_IMAGE_DEFAULTS.widthM),
        clampLayoutImageSize(overlay.heightM, GRID_LAYOUT_IMAGE_DEFAULTS.heightM)
      );
      uctx.restore();
    }
    _gridUnderlayDirty = false;
  }

  function drawGrid() {
    const w = canvas.width / dpr, h = canvas.height / dpr;
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = GRID_VIEW_BG;
    ctx.fillRect(0, 0, w, h);
    ctx.restore();
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.translate(state.panX, state.panY);
    ctx.scale(state.scale, state.scale);
    const maxX = GRID_COLS * CELL_SIZE, maxY = GRID_ROWS * CELL_SIZE;
    if (_gridUnderlayDirty) rebuildGridUnderlay();
    ctx.drawImage(_gridUnderlayCanvas, 0, 0, maxX, maxY);
    if (!state.showGrid) {
      ctx.restore();
      return;
    }
    const drawMinor = !(GRID_MINOR_GRID_MIN_SCALE > 0 && state.scale < GRID_MINOR_GRID_MIN_SCALE);
    const marginWorld = GRID_DRAW_VIEWPORT_MARGIN_CELLS * CELL_SIZE;
    const s = state.scale || 1;
    const minWx = (0 - state.panX) / s - marginWorld;
    const maxWx = (w - state.panX) / s + marginWorld;
    const minWy = (0 - state.panY) / s - marginWorld;
    const maxWy = (h - state.panY) / s + marginWorld;
    const cMin = Math.max(0, Math.floor(minWx / CELL_SIZE));
    const cMax = Math.min(GRID_COLS, Math.ceil(maxWx / CELL_SIZE));
    const rMin = Math.max(0, Math.floor(minWy / CELL_SIZE));
    const rMax = Math.min(GRID_ROWS, Math.ceil(maxWy / CELL_SIZE));
    for (let c = cMin; c <= cMax; c++) {
      const isMajor = (c % GRID_MAJOR_INTERVAL === 0);
      if (!isMajor && !drawMinor) continue;
      const x = c * CELL_SIZE;
      ctx.strokeStyle = isMajor
        ? ('rgba(' + GRID_MAJOR_LINE_RGB + ',' + GRID_MAJOR_LINE_OPACITY + ')')
        : ('rgba(' + GRID_MINOR_LINE_RGB + ',' + GRID_MINOR_LINE_OPACITY + ')');
      ctx.lineWidth = isMajor ? GRID_MAJOR_LINE_WIDTH : GRID_MINOR_LINE_WIDTH;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, maxY);
      ctx.stroke();
    }
    for (let r = rMin; r <= rMax; r++) {
      const isMajor = (r % GRID_MAJOR_INTERVAL === 0);
      if (!isMajor && !drawMinor) continue;
      const y = r * CELL_SIZE;
      ctx.strokeStyle = isMajor
        ? ('rgba(' + GRID_MAJOR_LINE_RGB + ',' + GRID_MAJOR_LINE_OPACITY + ')')
        : ('rgba(' + GRID_MINOR_LINE_RGB + ',' + GRID_MINOR_LINE_OPACITY + ')');
      ctx.lineWidth = isMajor ? GRID_MAJOR_LINE_WIDTH : GRID_MINOR_LINE_WIDTH;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(maxX, y);
      ctx.stroke();
    }
    ctx.fillStyle = '#aaa';
    ctx.font = '10px system-ui';
    ctx.fillText('0,0', 4, 2);
    const cx = (GRID_COLS * CELL_SIZE) / 2;
    const cy = (GRID_ROWS * CELL_SIZE) / 2;
    ctx.beginPath();
    ctx.fillStyle = '#ef4444';
    ctx.arc(cx, cy, CELL_SIZE * 0.15, 0, Math.PI * 2);
    ctx.fill();
    if (state.hoverCell != null) {
      const hc = state.hoverCell;
      const hx = hc.col * CELL_SIZE;
      const hy = hc.row * CELL_SIZE;
      ctx.beginPath();
      ctx.fillStyle = 'rgba(248, 113, 113, 0.45)';
      ctx.arc(hx, hy, CELL_SIZE * 0.2, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  }

  function drawPolygonHatch(points, strokeStyle, spacingPx) {
    if (!Array.isArray(points) || points.length < 3) return;
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    points.forEach(function(p) {
      minX = Math.min(minX, p[0]);
      maxX = Math.max(maxX, p[0]);
      minY = Math.min(minY, p[1]);
      maxY = Math.max(maxY, p[1]);
    });
    const span = Math.max(maxX - minX, maxY - minY);
    const pad = span + Math.max(40, spacingPx * 2);
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(points[0][0], points[0][1]);
    for (let i = 1; i < points.length; i++) ctx.lineTo(points[i][0], points[i][1]);
    ctx.closePath();
    ctx.clip();
    ctx.strokeStyle = strokeStyle;
    ctx.lineWidth = 1.2;
    ctx.setLineDash([]);
    for (let offset = minX - pad; offset <= maxX + pad; offset += spacingPx) {
      ctx.beginPath();
      ctx.moveTo(offset, maxY + pad);
      ctx.lineTo(offset + (maxY - minY) + pad, minY - pad);
      ctx.stroke();
    }
    ctx.restore();
  }
  function drawTerminals() {
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.translate(state.panX, state.panY);
    ctx.scale(state.scale, state.scale);
    state.terminals.forEach(term => {
      const isDrawingTerm = state.terminalDrawingId === term.id;
      if (term.vertices.length === 0 && !isDrawingTerm) return;
      const selected = state.selectedObject && state.selectedObject.type === 'terminal' && state.selectedObject.id === term.id;
      const buildingTheme = getBuildingTheme(term);
      const termPts = term.vertices.map(function(v) { return cellToPixel(v.col, v.row); });
      ctx.lineWidth = selected ? 3 : 2;
      ctx.strokeStyle = selected ? c2dObjectSelectedStroke() : buildingTheme.stroke;
      ctx.fillStyle = selected ? c2dObjectSelectedFill() : buildingTheme.fill;
      ctx.beginPath();
      for (let i = 0; i < termPts.length; i++) {
        const [x,y] = termPts[i];
        if (i === 0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
      }
      if (term.closed) {
        ctx.closePath();
        if (buildingTheme.fillEnabled) ctx.fill();
      }
      if (selected) {
        ctx.save();
        ctx.shadowColor = c2dObjectSelectedGlow();
        ctx.shadowBlur = c2dObjectSelectedGlowBlur();
        ctx.shadowOffsetX = 0;
        ctx.shadowOffsetY = 0;
      }
      ctx.stroke();
      if (selected) ctx.restore();
      if (term.closed && buildingTheme.hatch === 'diagonal' && buildingTheme.fillEnabled) {
        drawPolygonHatch(termPts, selected ? c2dObjectSelectedDashStroke() : buildingTheme.stroke, Math.max(10, CELL_SIZE * 0.6));
      }
      if (term.closed && term.vertices.length > 0) {
        let cx = 0, cy = 0;
        term.vertices.forEach(v => {
          const [px, py] = cellToPixel(v.col, v.row);
          cx += px; cy += py;
        });
        cx /= term.vertices.length;
        cy /= term.vertices.length;
        const label = term.name || term.id || 'Building';
        ctx.fillStyle = buildingTheme.labelFill;
        ctx.font = '12px system-ui';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(label, cx, cy);
      }
      term.vertices.forEach((v, i) => {
        const [x,y] = cellToPixel(v.col, v.row);
        const vertexSelected = isSelectedVertex('terminal', term.id, i);
        ctx.beginPath();
        ctx.fillStyle = vertexSelected ? '#f43f5e' : (i === 0 ? '#f97316' : '#e5e7eb');
        ctx.arc(x, y, layoutTerminalVertexRadiusPx(vertexSelected), 0, Math.PI*2);
        ctx.fill();
      });
      if (isDrawingTerm && state.layoutPathDrawPointer && term.vertices.length >= 1) {
        const ptr = state.layoutPathDrawPointer;
        const lastV = term.vertices[term.vertices.length - 1];
        const [lx, ly] = cellToPixel(lastV.col, lastV.row);
        if (ptr && ptr.length >= 2 && dist2([lx, ly], ptr) > 1e-6) {
          ctx.save();
          ctx.strokeStyle = 'rgba(250, 204, 21, 0.75)';
          ctx.setLineDash([4, 6]);
          ctx.lineWidth = 2;
          ctx.lineCap = 'round';
          ctx.beginPath();
          ctx.moveTo(lx, ly);
          ctx.lineTo(ptr[0], ptr[1]);
          ctx.stroke();
          ctx.restore();
        }
      }
    });
    ctx.restore();
  }

  function drawPBBs() {
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.translate(state.panX, state.panY);
    ctx.scale(state.scale, state.scale);
    state.pbbStands.forEach(pbb => {
      const x1 = Number(pbb.x1), y1 = Number(pbb.y1), x2 = Number(pbb.x2), y2 = Number(pbb.y2);
      if (!Number.isFinite(x1) || !Number.isFinite(y1) || !Number.isFinite(x2) || !Number.isFinite(y2)) return;
      rebuildPbbBridgeGeometry(pbb);
      const endSize = getStandSizeMeters(pbb.category || 'C');
      const sel = state.selectedObject && state.selectedObject.type === 'pbb' && state.selectedObject.id === pbb.id;
      const simOcc = state.hasSimulationResult && isStandOccupiedAtSimSec(pbb.id, state.simTimeSec);
      const bridges = Array.isArray(pbb.pbbBridges) ? pbb.pbbBridges : [];
      bridges.forEach(function(bridge, bridgeIdx) {
        const pts = Array.isArray(bridge.points) ? bridge.points : [];
        if (pts.length < 2) return;
        ctx.strokeStyle = sel ? c2dObjectSelectedStroke() : '#f97316';
        ctx.lineWidth = sel ? 3.5 : 2.5;
        if (sel) {
          ctx.save();
          ctx.shadowColor = c2dObjectSelectedGlow();
          ctx.shadowBlur = c2dObjectSelectedGlowBlur();
        }
        ctx.beginPath();
        ctx.moveTo(Number(pts[0].x) || 0, Number(pts[0].y) || 0);
        for (let pi = 1; pi < pts.length; pi++) ctx.lineTo(Number(pts[pi].x) || 0, Number(pts[pi].y) || 0);
        ctx.stroke();
        if (sel) ctx.restore();
        if (sel) {
          pts.forEach(function(pt, ptIdx) {
            const isBridgeVertexSelected = !!(state.selectedVertex && state.selectedVertex.type === 'pbbBridge' && state.selectedVertex.id === pbb.id && state.selectedVertex.bridgeIndex === bridgeIdx && state.selectedVertex.pointIndex === ptIdx);
            ctx.beginPath();
            ctx.fillStyle = isBridgeVertexSelected ? '#f43f5e' : '#fdba74';
            ctx.arc(Number(pt.x) || 0, Number(pt.y) || 0, isBridgeVertexSelected ? 4 : 3, 0, Math.PI * 2);
            ctx.fill();
          });
        }
      });
      const apronPt = getStandConnectionPx(pbb);
      const ex = apronPt[0], ey = apronPt[1];
      const angle = getPBBStandAngle(pbb);
      const rotationActive = !!(state.selectedVertex && state.selectedVertex.type === 'standRotation' && state.selectedVertex.id === pbb.id);
      const apronLinked = standHasApronTaxiwayLink(pbb.id);
      const idleFill = apronLinked ? 'rgba(22,163,74,0.18)' : 'rgba(107,114,128,0.22)';
      const idleStroke = apronLinked ? '#22c55e' : '#9ca3af';
      ctx.fillStyle = sel ? c2dObjectSelectedFill() : (simOcc ? c2dSimStandOccupiedFill() : idleFill);
      ctx.strokeStyle = sel ? c2dObjectSelectedStroke() : (simOcc ? c2dSimStandOccupiedStroke() : idleStroke);
      ctx.lineWidth = sel ? 2.5 : 1.5;
      ctx.save();
      ctx.translate(ex, ey);
      ctx.rotate(angle);
      ctx.beginPath();
      ctx.rect(-endSize/2, -endSize/2, endSize, endSize);
      ctx.fill();
      if (sel) {
        ctx.save();
        ctx.shadowColor = c2dObjectSelectedGlow();
        ctx.shadowBlur = c2dObjectSelectedGlowBlur();
      }
      ctx.stroke();
      if (sel) ctx.restore();
      const nameRaw = (pbb.name && pbb.name.trim()) ? pbb.name.trim() : String(state.pbbStands.indexOf(pbb) + 1);
      const labelPrefix = getStandCategoryMode(pbb) === 'aircraft' ? 'AC' : (pbb.category || 'C');
      const label = labelPrefix + ' / ' + nameRaw;
      const pad = 3;
      const tx = endSize / 2 - pad;
      const ty = -endSize / 2 + pad;
      ctx.fillStyle = apronLinked ? '#bbf7d0' : '#d1d5db';
      ctx.font = '8px system-ui';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'top';
