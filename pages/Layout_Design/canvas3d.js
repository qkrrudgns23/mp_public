          const p1 = cellToPixel(v1.col, v1.row), p2 = cellToPixel(v2.col, v2.row);
          const near = closestPointOnSegment(p1, p2, snappedPx);
          if (near) {
            const d2 = dist2(near, snappedPx);
            if (d2 < bestD2) { bestD2 = d2; bestEdge = { near, p1, p2 }; }
          }
        }
      });
      const maxD2 = (CELL_SIZE*1.0)**2;
      if (bestEdge && bestD2 < maxD2) {
        const nearPt = bestEdge.near;
        const ex = (nearPt && nearPt[0] != null) ? nearPt[0] : 0;
        const ey = (nearPt && nearPt[1] != null) ? nearPt[1] : 0;
        const [x1,y1]=bestEdge.p1, [x2,y2]=bestEdge.p2;
        let nx = -(y2-y1), ny = x2-x1;
        const len = Math.hypot(nx,ny) || 1; nx /= len; ny /= len;
        const toClickX = snappedPx[0] - ex, toClickY = snappedPx[1] - ey;
        if (nx * toClickX + ny * toClickY < 0) { nx *= -1; ny *= -1; }
        const category = document.getElementById('standCategory').value || 'C';
        const standSize = getStandSizeMeters(category);
        const minLen = standSize / 2 + 3;
        const lenMeters = Number(document.getElementById('pbbLength').value || 15);
        const lenPx = Math.max(isFinite(lenMeters) && lenMeters > 0 ? lenMeters : 15, minLen);
        const px2 = ex + nx * lenPx, py2 = ey + ny * lenPx;
        const preview = { x1: ex, y1: ey, x2: px2, y2: py2, category };
        const overlap = pbbStandOverlapsExisting(preview);
        state.previewPbb = { x1: ex, y1: ey, x2: px2, y2: py2, category: preview.category, overlap };
        scheduleDraw(); drewThisMove = true;
      } else {
        if (state.previewPbb) { state.previewPbb = null; scheduleDraw(); drewThisMove = true; }
      }
    } else {
      let clearedPreview = false;
      if (state.previewRemote) { state.previewRemote = null; clearedPreview = true; }
      if (state.previewPbb) { state.previewPbb = null; clearedPreview = true; }
      if (state.previewHoldingPoint) { state.previewHoldingPoint = null; clearedPreview = true; }
      if (clearedPreview) { scheduleDraw(); drewThisMove = true; }
    }
    if (flightTooltip && !state.isPanning) {
      let tipDone = false;
      if (state.hasSimulationResult && state.globalUpdateFresh) {
        let bestFlight = null;
        let bestD2 = (CELL_SIZE * FLIGHT_TOOLTIP_CF) ** 2;
        const tSec = state.simTimeSec;
        if (typeof prepareLazyTimelinesForCurrentSim === 'function') prepareLazyTimelinesForCurrentSim(tSec);
        state.flights.forEach(f => {
          const pose = getFlightPoseAtTimeForDraw(f, tSec);
          if (!pose || f.reg == null || !String(f.reg).trim()) return;
          const dx = pose.x - wx;
          const dy = pose.y - wy;
          const d2 = dx * dx + dy * dy;
          if (d2 < bestD2) { bestD2 = d2; bestFlight = f; }
        });
        if (bestFlight && bestFlight.reg) {
          flightTooltip.style.display = 'block';
          flightTooltip.textContent = String(bestFlight.reg).trim();
          flightTooltip.style.left = (ev.clientX + 12) + 'px';
          flightTooltip.style.top = (ev.clientY + 12) + 'px';
          tipDone = true;
        }
      }
      if (!tipDone) {
        const hit = hitTest(wx, wy);
        if (hit && hit.obj) {
          const name = (hit.obj.name != null && String(hit.obj.name).trim()) ? String(hit.obj.name).trim() : (hit.type === 'terminal' ? 'Building' : hit.type === 'pbb' ? 'Contact Stand' : hit.type === 'remote' ? 'Remote Stand' : hit.type === 'holdingPoint' ? holdingPointKindDisplayLabel(hit.obj.hpKind) : hit.type === 'taxiway' ? (hit.obj.name || 'Path') : hit.type === 'apronLink' ? (hit.obj.name || 'Apron Taxiway') : hit.type);
          flightTooltip.style.display = 'block';
          flightTooltip.textContent = name;
          flightTooltip.style.left = (ev.clientX + 12) + 'px';
          flightTooltip.style.top = (ev.clientY + 12) + 'px';
        } else {
          flightTooltip.style.display = 'none';
        }
      }
    }
    if (hoverChanged && !drewThisMove) { scheduleDraw(); drewThisMove = true; }
  });
  container.addEventListener('mouseleave', function() {
    state.dragStart = null;
    state.isPanning = false;
    state.dragStandRotation = null;
    state.dragPbbBridgeVertex = null;
    state.dragStandConnection = null;
    state.hoverCell = null;
    state.previewPbb = null;
    state.previewRemote = null;
    state.previewHoldingPoint = null;
    state.apronLinkPointerWorld = null;
    flushDrawNow();
  });
  container.addEventListener('dblclick', function(ev) {
    if (ev.button !== 0) return;
    const rect = canvas.getBoundingClientRect();
    const sx = ev.clientX - rect.left, sy = ev.clientY - rect.top;
    const [wx, wy] = screenToWorld(sx, sy);
    if (insertSelectedVertexAt(wx, wy, !!ev.shiftKey)) {
      ev.preventDefault();
    }
  });
  function hitTestPbbEnd(wx, wy) {
    const maxD2 = (CELL_SIZE * HIT_PBB_END_CF) ** 2;
    const cands = [];
    state.pbbStands.forEach(pbb => {
      const pt = getStandConnectionPx(pbb);
      cands.push({ id: pbb.id, kind: 'pbb', x: pt[0], y: pt[1] });
    });
    state.remoteStands.forEach(st => {
      const [cx, cy] = getRemoteStandCenterPx(st);
      cands.push({ id: st.id, kind: 'remote', x: cx, y: cy });
    });
    const best = findNearestItem(cands, c => [c.x, c.y], wx, wy, maxD2);
    return best || null;
  }

  function hitTestAnyTaxiwayVertex(wx, wy) {
    const click = [wx, wy];
    const maxD2 = (CELL_SIZE * TRY_PBB_MAX_EDGE_CF) ** 2;
    let best = null;
    let bestD2 = maxD2;
    state.taxiways.forEach(tw => {
      if (!tw.vertices || tw.vertices.length < 2) return;
      for (let i = 0; i < tw.vertices.length - 1; i++) {
        const [x1, y1] = cellToPixel(tw.vertices[i].col, tw.vertices[i].row);
        const [x2, y2] = cellToPixel(tw.vertices[i+1].col, tw.vertices[i+1].row);
        const near = closestPointOnSegment([x1, y1], [x2, y2], click);
        if (!near) continue;
        const d2 = dist2(near, click);
        if (d2 < bestD2) {
          bestD2 = d2;
          best = { taxiwayId: tw.id, x: near[0], y: near[1] };
        }
      }
    });
    return best;
  }

  container.addEventListener('mouseup', function(ev) {
    if (ev.button !== 0) return;
    const wasPanning = !!state.isPanning;
    flushDrawNow();
    state.isPanning = false;
    if (state.dragVertex) {
      state.dragVertex = null;
      return;
    }
    if (state.dragTaxiwayVertex) {
      const tw = state.taxiways.find(t => t.id === state.dragTaxiwayVertex.taxiwayId);
      if (tw && typeof syncStartEndFromVertices === 'function') syncStartEndFromVertices(tw);
      state.dragTaxiwayVertex = null;
      if (typeof syncPanelFromState === 'function') syncPanelFromState();
      if (typeof updateObjectInfo === 'function') updateObjectInfo();
      if (typeof redrawLayoutAfterEdit === 'function') redrawLayoutAfterEdit();
      else {
        if (typeof updateAllFlightPaths === 'function') updateAllFlightPaths();
        if (scene3d) update3DScene();
        draw();
      }
      return;
    }
    if (state.dragStandRotation) {
      state.dragStandRotation = null;
      if (typeof syncPanelFromState === 'function') syncPanelFromState();
      if (typeof updateObjectInfo === 'function') updateObjectInfo();
      if (typeof redrawLayoutAfterEdit === 'function') redrawLayoutAfterEdit();
      else {
        if (typeof updateAllFlightPaths === 'function') updateAllFlightPaths();
        if (scene3d) update3DScene();
        draw();
      }
      return;
    }
    if (state.dragPbbBridgeVertex) {
      state.dragPbbBridgeVertex = null;
      updateObjectInfo();
      draw();
      return;
    }
    if (state.dragStandConnection) {
      state.dragStandConnection = null;
      updateObjectInfo();
      if (typeof redrawLayoutAfterEdit === 'function') redrawLayoutAfterEdit();
      else {
        if (typeof updateAllFlightPaths === 'function') updateAllFlightPaths(); else draw();
        if (scene3d) update3DScene();
      }
      return;
    }
    if (state.dragApronLinkVertex) {
      state.dragApronLinkVertex = null;
      if (typeof updateObjectInfo === 'function') updateObjectInfo();
      if (typeof redrawLayoutAfterEdit === 'function') redrawLayoutAfterEdit();
      else {
        if (typeof updateAllFlightPaths === 'function') updateAllFlightPaths();
        if (scene3d) update3DScene();
        draw();
      }
      return;
    }
    const rect = canvas.getBoundingClientRect();
    const sx = ev.clientX - rect.left, sy = ev.clientY - rect.top;
    const [wx, wy] = screenToWorld(sx, sy);
    const placePx = worldPointToPixel(wx, wy, !!ev.shiftKey);
    const mode = settingModeSelect.value;
    const inStandDrawingMode = (mode === 'pbb' && state.pbbDrawing) || (mode === 'remote' && state.remoteDrawing) || (mode === 'holdingPoint' && state.holdingPointDrawing);
    if (!state.dragStart && !inStandDrawingMode) { state.dragStart = null; return; }
    if (handlePbbOrRemoteMouseUp2D(mode, placePx[0], placePx[1])) {
      state.dragStart = null;
      return;
    }
    if (!state.dragStart) return;
    if (!wasPanning) {
      const mode = settingModeSelect.value;
      if (mode === 'edge') {
        rebuildDerivedGraphEdges();
        const eh = hitTestLayoutGraphEdge(wx, wy);
        if (eh) {
          state.selectedObject = { type: 'layoutEdge', id: eh.id, obj: eh };
        } else {
          state.selectedObject = null;
        }
        state.flightPathRevealFlightId = null;
        syncPanelFromState();
        updateObjectInfo();
        draw();
        if (typeof syncAllocGanttSelectionHighlight === 'function') syncAllocGanttSelectionHighlight();
        state.dragStart = null;
        return;
      }
      const hit = hitTest(wx, wy);
      if (mode === 'apronTaxiway' && state.apronLinkDrawing) {
        const pbbHit = hitTestPbbEnd(wx, wy);
        const twHit = hitTestAnyTaxiwayVertex(wx, wy);
        const endpoint = pbbHit ? { kind: pbbHit.kind, standId: pbbHit.id, x: pbbHit.x, y: pbbHit.y } :
                          (twHit ? { kind: 'taxiway', taxiwayId: twHit.taxiwayId, x: twHit.x, y: twHit.y } : null);
        if (endpoint) {
          if (!state.apronLinkTemp) {
            state.apronLinkTemp = endpoint;
            state.apronLinkMidpoints = [];
          } else {
            const first = state.apronLinkTemp;
            if (first.kind !== endpoint.kind) {
              let standId, taxiwayId, tx, ty, midVertices;
              if (first.kind === 'taxiway') {
                taxiwayId = first.taxiwayId;
                standId = endpoint.standId;
                tx = first.x;
                ty = first.y;
                midVertices = (state.apronLinkMidpoints || []).slice().reverse();
              } else {
                taxiwayId = endpoint.taxiwayId;
                standId = first.standId;
                tx = endpoint.x;
                ty = endpoint.y;
                midVertices = (state.apronLinkMidpoints || []).slice();
              }
              if (standId && taxiwayId) {
                const newId = id();
                const inputName = document.getElementById('apronLinkName');
                const linkName = (inputName && String(inputName.value).trim()) || getApronLinkDefaultName(newId);
                if (findDuplicateLayoutName('apronLink', newId, linkName)) {
                  alertDuplicateLayoutName();
                } else {
                  pushUndo();
                  const linkRec = { id: newId, name: linkName, pbbId: standId, taxiwayId, tx, ty };
                  if (midVertices && midVertices.length) linkRec.midVertices = midVertices;
                  state.apronLinks.push(linkRec);
                  syncPanelFromState();
                  if (typeof redrawLayoutAfterEdit === 'function') redrawLayoutAfterEdit();
                  else {
                    if (typeof updateAllFlightPaths === 'function') updateAllFlightPaths();
                    if (scene3d) update3DScene();
                  }
                }
              }
            }
            state.apronLinkTemp = null;
            state.apronLinkMidpoints = [];
            state.apronLinkPointerWorld = null;
          }
          draw();
        } else if (state.apronLinkTemp) {
          const [col, row] = pixelToCell(wx, wy);
          if (col >= 0 && row >= 0 && col <= GRID_COLS && row <= GRID_ROWS) {
            const last = state.apronLinkMidpoints[state.apronLinkMidpoints.length - 1];
            if (!last || last.col !== col || last.row !== row) {
              state.apronLinkMidpoints.push({ col, row });
            }
          }
          draw();
        }
      } else if (hit) {
        state.flightPathRevealFlightId = null;
        state.selectedObject = hit;
        if (hit.type === 'terminal') state.currentTerminalId = hit.id;
        const sm = settingModeValueForHit(hit);
        if (sm) settingModeSelect.value = sm;
        if (hit.type === 'flight' && typeof switchToTab === 'function') switchToTab('flight');
        if (typeof syncSettingsPaneToMode === 'function') syncSettingsPaneToMode();
        syncPanelFromState();
        renderObjectList();
        updateObjectInfo();
        draw();
        if (typeof syncAllocGanttSelectionHighlight === 'function') syncAllocGanttSelectionHighlight();
      } else {
        const pt = worldPointToCellPoint(wx, wy, !!ev.shiftKey);
        const col = pt.col, row = pt.row;
        if (col < 0 || row < 0 || col > GRID_COLS || row > GRID_ROWS) { state.dragStart = null; return; }
        if (mode === 'terminal') {
          if (state.terminalDrawingId) {
            let term = state.terminals.find(t => t.id === state.terminalDrawingId);
            if (!term) {
              state.terminalDrawingId = null;
              state.layoutPathDrawPointer = null;
            } else {
              const pt = { col, row };
              if (term.vertices.length === 0) {
                pushUndo();
                term.vertices.push(pt);
              } else {
                const [fx,fy] = cellToPixel(term.vertices[0].col, term.vertices[0].row);
                const d2 = dist2([fx,fy], cellToPixel(col, row));
                if (d2 < (CELL_SIZE * TERM_CLOSE_POLY_CF) ** 2 && term.vertices.length >= 3) {
                  term.closed = true;
                  state.terminalDrawingId = null;
                  state.layoutPathDrawPointer = null;
                  syncPanelFromState();
                } else {
                  const last = term.vertices[term.vertices.length-1];
                  if (last.col !== col || last.row !== row) { pushUndo(); term.vertices.push(pt); }
                }
              }
              draw();
            }
          }
        } else if (isPathLayoutMode(mode)) {
          if (state.taxiwayDrawingId) {
            const tw = state.taxiways.find(t => t.id === state.taxiwayDrawingId);
            if (tw) {
              const pt = { col, row };
              const last = tw.vertices[tw.vertices.length - 1];
              if (!last || last.col !== col || last.row !== row) {
                if (tw.pathType === 'runway' && tw.vertices.length >= 2) return;
                pushUndo();
                tw.vertices.push(pt);
                if (typeof syncStartEndFromVertices === 'function') syncStartEndFromVertices(tw);
                if (tw.pathType === 'runway' && tw.vertices.length >= 2) {
                  state.taxiwayDrawingId = null;
                  state.layoutPathDrawPointer = null;
                  syncPanelFromState();
                }
                if (typeof redrawLayoutAfterEdit === 'function') redrawLayoutAfterEdit();
                else if (typeof updateAllFlightPaths === 'function') updateAllFlightPaths(); else draw();
              }
            }
          }
        } else if (mode === 'pbb') {
          if (tryPlacePbbAt(wx, wy)) {
            syncPanelFromState();
            draw();
          }
        } else if (mode === 'remote' && state.remoteDrawing) {
          const prev = state.previewRemote;
          if (prev && !prev.overlap && tryPlaceRemoteAt(prev.x, prev.y)) {
            syncPanelFromState();
            draw();
          }
        }
      }
    }
    state.dragStart = null;
  });
  let scene3d = null, camera3d = null, renderer3d = null, controls3d = null, grid3DMapper = null, raycaster3d = null, mouse3d = null, groundPlane3d = null, gridGroup3d = null;
  let mouse3dDown = null;
  const view3dContainer = document.getElementById('view3d-container');
  document.getElementById('btnView2D').classList.add('active');
  document.getElementById('btnView2D').addEventListener('click', function() {
    document.getElementById('btnView2D').classList.add('active');
    document.getElementById('btnView3D').classList.remove('active');
    document.getElementById('canvas-container').style.display = 'block';
    view3dContainer.classList.remove('active');
    if (renderer3d) renderer3d.domElement.style.display = 'none';
    requestAnimationFrame(function() {
      if (typeof resizeCanvas === 'function') resizeCanvas();
    });
  });
  document.getElementById('btnView3D').addEventListener('click', function() {
    document.getElementById('btnView3D').classList.add('active');
    document.getElementById('btnView2D').classList.remove('active');
    document.getElementById('canvas-container').style.display = 'none';
    view3dContainer.classList.add('active');
    init3D();
    animate3D();
  });

  function reset3DView() {
    if (!camera3d) return;
    const halfW = (GRID_COLS * CELL_SIZE) / 2;
    const halfH = (GRID_ROWS * CELL_SIZE) / 2;
    const maxDim = Math.max(halfW, halfH);
    camera3d.position.set(maxDim * 1.2, maxDim * 1.2, maxDim * 1.2);
    camera3d.lookAt(0, 0, 0);
    if (controls3d) {
      controls3d.target.set(0, 0, 0);
      controls3d.update();
    }
  }

  if (resetViewBtn) {
    resetViewBtn.addEventListener('click', function() {
      try {
        resizeCanvas();
        if (view3dContainer.classList.contains('active')) reset3DView();
        else reset2DView();
        try { draw(); } catch(e) {}
        if (typeof update3DScene === 'function') update3DScene();
      } catch (e) { console.error('Fit button error:', e); }
    });
  }
  if (gridToggleBtn) {
    syncGridToggleButton();
    gridToggleBtn.addEventListener('click', function() {
      state.showGrid = !state.showGrid;
      syncGridToggleButton();
      draw();
    });
  }
  if (imageToggleBtn) {
    syncImageToggleButton();
    imageToggleBtn.addEventListener('click', function() {
      state.showImage = !state.showImage;
      syncImageToggleButton();
      invalidateGridUnderlay();
      draw();
    });
  }
  class Grid3DMapper {
    constructor(cols, rows, cellSize) {
      this.cols = cols;
      this.rows = rows;
      this.cellSize = cellSize;
      this.ox = (cols * cellSize) / 2;
      this.oz = (rows * cellSize) / 2;
    }
    pixelToWorldXZ(x, y) {
      return { x: this.ox - x, z: this.oz - y };
    }
    cellToWorld(col, row, height) {
      const [px, py] = cellToPixel(col, row);
      const p = this.pixelToWorldXZ(px, py);
      return new THREE.Vector3(p.x, height, p.z);
    }
    worldFromPixel(x, y, height) {
      const p = this.pixelToWorldXZ(x, y);
      return new THREE.Vector3(p.x, height, p.z);
    }
    shapeFromCell(col, row) {
      const [px, py] = cellToPixel(col, row);
      return { x: this.ox - px, y: py - this.oz };
    }
    worldToPixel(xWorld, zWorld) {
      return { x: this.ox - xWorld, y: this.oz - zWorld };
    }
    worldToCell(xWorld, zWorld) {
      const p = this.worldToPixel(xWorld, zWorld);
      let col = Math.round(p.x / this.cellSize);
      let row = Math.round(p.y / this.cellSize);
      col = Math.max(0, Math.min(this.cols, col));
      row = Math.max(0, Math.min(this.rows, row));
      return [col, row];
    }
  }

  function init3D() {
    if (renderer3d) { renderer3d.domElement.style.display = 'block'; update3DScene(); return; }
    const w = view3dContainer.clientWidth, h = view3dContainer.clientHeight;
    scene3d = new THREE.Scene();
    scene3d.background = new THREE.Color(GRID_VIEW_BG);
    gridGroup3d = new THREE.Group();
    scene3d.add(gridGroup3d);
    camera3d = new THREE.PerspectiveCamera(50, w/h, 1, 100000);
    const halfW = (GRID_COLS * CELL_SIZE) / 2, halfH = (GRID_ROWS * CELL_SIZE) / 2;
    const maxDim = Math.max(halfW, halfH);
    camera3d.position.set(maxDim * 1.2, maxDim * 1.2, maxDim * 1.2);
    camera3d.lookAt(0, 0, 0);
    const axisLen = CELL_SIZE * 8;
    const axisOrigin = new THREE.Vector3(-maxDim, 0, -maxDim);
    function addAxis(toVec, color) {
      const pts = [axisOrigin, axisOrigin.clone().add(toVec)];
      const geo = new THREE.BufferGeometry().setFromPoints(pts);
      const mat = new THREE.LineBasicMaterial({ color });
      const line = new THREE.Line(geo, mat);
      gridGroup3d.add(line);
    }
    addAxis(new THREE.Vector3(axisLen, 0, 0), 0xef4444);
    addAxis(new THREE.Vector3(0, 0, axisLen), 0x22c55e);
    addAxis(new THREE.Vector3(0, axisLen, 0), 0x7c6af7);
    function createAxisLabel(text, color, endVec) {
      const size = 128;
      const canvasLabel = document.createElement('canvas');
      canvasLabel.width = size;
      canvasLabel.height = size;
      const g = canvasLabel.getContext('2d');
      g.clearRect(0, 0, size, size);
      g.font = 'bold 72px system-ui';
      g.fillStyle = color;
      g.textAlign = 'center';
      g.textBaseline = 'middle';
      g.fillText(text, size / 2, size / 2);
      const tex = new THREE.CanvasTexture(canvasLabel);
      const mat = new THREE.SpriteMaterial({ map: tex, transparent: true });
      const sprite = new THREE.Sprite(mat);
      const s = CELL_SIZE * 3;
      sprite.scale.set(s, s, 1);
      sprite.position.copy(axisOrigin.clone().add(endVec));
      gridGroup3d.add(sprite);
    }
    createAxisLabel('x', '#ef4444', new THREE.Vector3(axisLen * 1.1, 0, 0));
    createAxisLabel('y', '#22c55e', new THREE.Vector3(0, 0, axisLen * 1.1));
    createAxisLabel('z', '#7c6af7', new THREE.Vector3(0, axisLen * 1.1, 0));
    grid3DMapper = new Grid3DMapper(GRID_COLS, GRID_ROWS, CELL_SIZE);
    renderer3d = new THREE.WebGLRenderer({ antialias: true });
    renderer3d.setSize(w, h);
    renderer3d.setPixelRatio(window.devicePixelRatio || 1);
    view3dContainer.appendChild(renderer3d.domElement);
    controls3d = new THREE.OrbitControls(camera3d, renderer3d.domElement);
    controls3d.enableDamping = true;
    controls3d.dampingFactor = 0.05;
    raycaster3d = new THREE.Raycaster();
    mouse3d = new THREE.Vector2();
    groundPlane3d = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
    const dom3d = renderer3d.domElement;
    function getHitPoint(ev) {
      const rect = dom3d.getBoundingClientRect();
      const ndcX = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
      const ndcY = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
      mouse3d.set(ndcX, ndcY);
      raycaster3d.setFromCamera(mouse3d, camera3d);
      const hit = new THREE.Vector3();
      return raycaster3d.ray.intersectPlane(groundPlane3d, hit) ? hit : null;
    }
    dom3d.addEventListener('mousedown', function(ev) {
      if (ev.button === 0) mouse3dDown = { x: ev.clientX, y: ev.clientY };
    });
    dom3d.addEventListener('mouseup', function(ev) {
      if (ev.button !== 0 || !mouse3dDown) return;
      const dx = ev.clientX - mouse3dDown.x, dy = ev.clientY - mouse3dDown.y;
      if (dx*dx + dy*dy > 25) { mouse3dDown = null; return; }
      mouse3dDown = null;
      const hit = getHitPoint(ev);
      if (!hit || !grid3DMapper) return;
      const mode = settingModeSelect.value;
      const p = grid3DMapper.worldToPixel(hit.x, hit.z);
      const wx = p.x, wy = p.y;
      const [col, row] = grid3DMapper.worldToCell(hit.x, hit.z);
      tryCommitStandPlacement3D(mode, wx, wy, col, row);
    });
    const step = CELL_SIZE;
    const faintLines = [];
    const majorLines = [];
    let kx = 0;
    for (let x = -maxDim; x <= maxDim; x += step, kx++) {
      const pts = [new THREE.Vector3(x, 0, -maxDim), new THREE.Vector3(x, 0, maxDim)];
      if (kx % GRID_MAJOR_INTERVAL === 0) majorLines.push.apply(majorLines, pts);
      else faintLines.push.apply(faintLines, pts);
    }
    let kz = 0;
    for (let z = -maxDim; z <= maxDim; z += step, kz++) {
      const pts = [new THREE.Vector3(-maxDim, 0, z), new THREE.Vector3(maxDim, 0, z)];
      if (kz % GRID_MAJOR_INTERVAL === 0) majorLines.push.apply(majorLines, pts);
      else faintLines.push.apply(faintLines, pts);
    }
    if (faintLines.length) {
      const faintGeo = new THREE.BufferGeometry().setFromPoints(faintLines);
      const faintMat = new THREE.LineBasicMaterial({
        color: 0xd4d4d4,
        transparent: true,
        opacity: 0.2,
        depthTest: false
      });
      gridGroup3d.add(new THREE.LineSegments(faintGeo, faintMat));
    }
    if (majorLines.length) {
      const majorGeo = new THREE.BufferGeometry().setFromPoints(majorLines);
      const majorMat = new THREE.LineBasicMaterial({
        color: 0xffffff,
        transparent: true,
        opacity: 0.35,
        depthTest: false
      });
      gridGroup3d.add(new THREE.LineSegments(majorGeo, majorMat));
    }
    update3DScene();
  }

  function update3DScene() {
    if (!scene3d) return;
    while (scene3d.children.length > 1) scene3d.remove(scene3d.children[scene3d.children.length - 1]);
    if (!grid3DMapper) grid3DMapper = new Grid3DMapper(GRID_COLS, GRID_ROWS, CELL_SIZE);
  }

  function animate3D() {
    if (!renderer3d || !view3dContainer.classList.contains('active')) return;
    requestAnimationFrame(animate3D);
    if (controls3d) controls3d.update();
    if (renderer3d && scene3d && camera3d) renderer3d.render(scene3d, camera3d);
  }

  container.addEventListener('wheel', function(ev) {
    ev.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const mx = ev.clientX - rect.left, my = ev.clientY - rect.top;
    const wx = (mx - state.panX) / state.scale, wy = (my - state.panY) / state.scale;
    const factor = 1 - ev.deltaY * 0.002;
    state.scale *= factor;
    state.scale = Math.max(CANVAS_MIN_ZOOM, Math.min(CANVAS_MAX_ZOOM, state.scale));
    state.panX = mx - wx * state.scale;
    state.panY = my - wy * state.scale;
    try { draw(); } catch(e) {}
  }, { passive: false });

  window.addEventListener('resize', function() {
    resizeCanvas();
    if (renderer3d && view3dContainer.classList.contains('active')) {
      const w = view3dContainer.clientWidth, h = view3dContainer.clientHeight;
      camera3d.aspect = w / h;
      camera3d.updateProjectionMatrix();
      renderer3d.setSize(w, h);
    }
  });
  try { applyInitialLayoutFromJson(); } catch(applyErr) { console.error('Layout apply failed:', applyErr); }
  updateLayoutNameBar(INITIAL_LAYOUT_DISPLAY_NAME || 'default_layout');
  resizeCanvas();
  reset2DView();
  syncPanelFromState();
  if (typeof draw === 'function') draw();
  if (typeof update3DScene === 'function') update3DScene();
  if (typeof renderKpiDashboard === 'function') renderKpiDashboard('Initial load');
})();
