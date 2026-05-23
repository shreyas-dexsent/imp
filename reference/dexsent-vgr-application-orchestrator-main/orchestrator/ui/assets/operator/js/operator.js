
(function () {
  const TASK_TYPE_ALIASES = {
    pick_place_demo: "pick_place_demo",
    bin_picking: "bin_picking",
    vim303_pick_place: "pick_place_demo",
    follow_object: "follow_object",
    pallatizing: "pallatizing",
    palletizing: "pallatizing",
    dummy_testing: "dummy_testing",
    dummy: "dummy_testing",
  };

  const TASK_TYPE_LABELS = {
    pick_place_demo: "Pick and Place",
    bin_picking: "Bin Picking",
    follow_object: "Object Tracking",
    pallatizing: "Pallatizing",
    dummy_testing: "Dummy Testing",
  };

  const BASE_TASK_TYPES = ["pick_place_demo", "bin_picking", "follow_object", "pallatizing", "dummy_testing"];

  const VISION_MODULE_OPTIONS = [
    ["Template SIFT", "tamplate_matching_sift"],
    ["Opt SIFT", "opt_sift"],
    ["MegaPose Bin Picking", "megapose_bin_picking"],
  ];

  const PROCESS_MODE_OPTIONS = [
    ["continuous", "continuous"],
    ["single", "single"],
    ["trigger_only", "trigger_only"],
  ];

  const ROBOT_PROFILE_OPTIONS = [
    ["slow", "slow"],
    ["normal", "normal"],
    ["fast", "fast"],
  ];

  const ACTION_BLOCK_TYPES = new Set([
    "step_capture",
    "step_pick",
    "step_intermediate_pose",
    "step_place",
    "step_move_pose",
    "step_delay",
    "step_track",
    "step_repeat",
    "step_wait_for_object",
    "step_gripper_action",
    "step_switch_profile",
    "step_config_vision_core",
    "step_config_vision_quality",
    "step_config_pick_profile",
    "step_config_follow_profile",
    "step_config_pallatizing_profile",
    "step_set_task_type",
    "control_while_true",
  ]);

  const state = {
    stations: [],
    assets: [],
    tasks: [],
    poses: [],
    objects: [],
    cameraIds: [],
    taskTypes: [...BASE_TASK_TYPES],
    flow: [],
    currentStationId: "",
    currentAssetId: "",
    currentTaskId: "",
    currentTaskType: "pick_place_demo",
    currentTaskPayload: {},
    currentRunId: "",
    dummyProgram: [],
    statusTimer: null,
    runTimer: null,
    workspace: null,
    blocklyReady: false,
    resizeHandler: null,
    boundVisionTransport: false,
  };

  function byId(id) {
    return document.getElementById(id);
  }

  function canonicalTaskType(value) {
    const raw = String(value || "").trim().toLowerCase();
    return TASK_TYPE_ALIASES[raw] || raw || "pick_place_demo";
  }

  function taskTypeLabel(value) {
    const canonical = canonicalTaskType(value);
    return TASK_TYPE_LABELS[canonical] || canonical || "Pick and Place";
  }

  function defaultTaskNameForType(value) {
    const canonical = canonicalTaskType(value);
    const names = {
      pick_place_demo: "pick_place",
      bin_picking: "bin_picking",
      follow_object: "tracking",
      pallatizing: "pallatizing",
      dummy_testing: "dummy_testing",
    };
    return names[canonical] || canonical || "operator-task";
  }

  function sanitize(v) {
    return String(v == null ? "" : v)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/\"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function toSafeNumber(value, fallback, minValue) {
    const n = Number(value);
    if (!Number.isFinite(n)) return fallback;
    if (typeof minValue === "number") return Math.max(minValue, n);
    return n;
  }

  function toSafeInteger(value, fallback, minValue) {
    const n = Number(value);
    if (!Number.isFinite(n)) return fallback;
    const rounded = Math.round(n);
    if (typeof minValue === "number") return Math.max(minValue, rounded);
    return rounded;
  }

  function fieldToBool(block, field) {
    return String(block.getFieldValue(field)).toUpperCase() === "TRUE";
  }

  function boolToField(value) {
    return value ? "TRUE" : "FALSE";
  }

  function safeSetField(block, field, value) {
    if (!block) return;
    try {
      block.setFieldValue(String(value == null ? "" : value), field);
    } catch (_) {
      // Ignore invalid dropdown values.
    }
  }

  function assetId(item) {
    if (!item || typeof item !== "object") return "";
    return String(item.asset_id || item.process_id || "").trim();
  }

  function setLed(id, mode) {
    const el = byId(id);
    if (!el) return;
    el.classList.remove("on", "off", "error");
    el.classList.add(mode || "off");
  }

  function setText(id, text) {
    const el = byId(id);
    if (el) el.textContent = text;
  }

  function infoBox(text) {
    setText("uiInfo", text || "");
  }

  function selectedAsset() {
    return state.assets.find((a) => assetId(a) === state.currentAssetId) || null;
  }

  function selectedTask() {
    return state.tasks.find((t) => t.task_id === state.currentTaskId) || null;
  }

  function fillSelect(id, items, getValue, getLabel, selectedValue) {
    const sel = byId(id);
    if (!sel) return;
    sel.innerHTML = "";
    (items || []).forEach((item) => {
      const opt = document.createElement("option");
      opt.value = getValue(item);
      opt.textContent = getLabel(item);
      sel.appendChild(opt);
    });
    if (selectedValue && [...sel.options].some((o) => o.value === selectedValue)) {
      sel.value = selectedValue;
      return;
    }
    if (sel.options.length) {
      sel.selectedIndex = 0;
    }
  }

  function ensureTaskTypePresent(taskType) {
    const canonical = canonicalTaskType(taskType);
    if (!state.taskTypes.includes(canonical)) {
      state.taskTypes.push(canonical);
      state.taskTypes.sort();
    }
  }

  function refreshTaskTypeSelect() {
    fillSelect(
      "taskTypeSelect",
      state.taskTypes,
      (t) => t,
      (t) => taskTypeLabel(t),
      state.currentTaskType
    );
  }

  function setTaskType(value, options) {
    const opts = options || {};
    const canonical = canonicalTaskType(value);
    ensureTaskTypePresent(canonical);
    state.currentTaskType = canonical;
    const sel = byId("taskTypeSelect");
    if (sel) sel.value = canonical;
    if (state.workspace && !opts.skipToolboxRefresh) {
      updateToolboxForTaskType();
    }
    if (opts.userSelected) {
      infoBox(`${taskTypeLabel(canonical)} profile selected. Save flow to persist task type.`);
    }
    renderTaskParamsPanel();
  }

  function optionsFromValues(values, fallbackLabel) {
    const arr = (values || []).map((v) => String(v || "").trim()).filter(Boolean);
    if (!arr.length) return [[fallbackLabel || "(none)", ""]];
    return arr.map((v) => [v, v]);
  }

  function cameraOptions() {
    return optionsFromValues(state.cameraIds, "(any camera)");
  }

  function objectOptions() {
    return optionsFromValues(
      state.objects.map((o) => o.object_id),
      "(no products)"
    );
  }

  function poseOptions() {
    return optionsFromValues(
      state.poses.map((p) => p.name),
      "(no poses)"
    );
  }

  function selectedPoseNames() {
    return state.poses.map((p) => String(p.name || "").trim()).filter(Boolean);
  }

  function fillDummyPoseSelect() {
    fillSelect(
      "dummyPoseSelect",
      selectedPoseNames(),
      (p) => p,
      (p) => p,
      ((byId("dummyPoseSelect") && byId("dummyPoseSelect").value) || "")
    );
  }

  function isDummyTestingActive() {
    return canonicalTaskType(state.currentTaskType) === "dummy_testing";
  }

  function dummyProgramFromFlow(flow) {
    const source = Array.isArray(flow) ? flow : [];
    const firstLoop = source.find((step) => normalizeFlowStepType(step && step.type) === "while_true");
    const steps = firstLoop && Array.isArray(firstLoop.body) ? firstLoop.body : source;
    return (steps || [])
      .filter((step) => {
        const type = normalizeFlowStepType(step && step.type);
        return ["move_pose", "intermediate_pose", "place", "delay", "gripper_action"].includes(type);
      })
      .map((step) => ({ ...step, type: normalizeFlowStepType(step.type) }));
  }

  function setDummyProgram(program) {
    state.dummyProgram = Array.isArray(program) ? program.map((step) => ({ ...step })) : [];
    renderDummyProgram();
  }

  function renderDummyProgram() {
    const host = byId("dummyPoseSequence");
    if (!host) return;
    if (!state.dummyProgram.length) {
      host.innerHTML = '<div class="hint">No pose sequence selected.</div>';
      return;
    }
    host.innerHTML = state.dummyProgram
      .map((step, idx) => {
        const type = normalizeFlowStepType(step && step.type);
        const pose = sanitize(step.pose_name || (type === "delay" ? "Delay" : "-"));
        const profile = sanitize(step.profile || "");
        return `
          <div class="dummy-sequence-row" data-index="${idx}">
            <span class="dummy-sequence-index">${idx + 1}</span>
            <span class="dummy-sequence-pose">${pose}</span>
            <span class="dummy-sequence-profile">${profile}</span>
            <span class="dummy-sequence-actions">
              <button class="ghost" type="button" data-action="up" data-index="${idx}">Up</button>
              <button class="ghost" type="button" data-action="down" data-index="${idx}">Down</button>
              <button class="danger" type="button" data-action="remove" data-index="${idx}">X</button>
            </span>
          </div>
        `;
      })
      .join("");
  }

  function renderTaskParamsPanel() {
    const panel = byId("dummyTestingParamsPanel");
    if (!panel) return;
    const active = isDummyTestingActive();
    panel.hidden = !active;
    if (!active) return;
    fillDummyPoseSelect();
    const cfg =
      state.currentTaskPayload &&
      state.currentTaskPayload.dummy_testing &&
      typeof state.currentTaskPayload.dummy_testing === "object"
        ? state.currentTaskPayload.dummy_testing
        : {};
    const loopInput = byId("dummyLoopIterations");
    if (loopInput && !String(loopInput.value || "").trim()) {
      loopInput.value = toSafeInteger(cfg.max_loop_iterations, 1, 1);
    }
    renderDummyProgram();
  }

  function addDummyPoseStep() {
    const poseName = (byId("dummyPoseSelect") && byId("dummyPoseSelect").value) || "";
    if (!poseName) {
      infoBox("Select a saved pose first.");
      return;
    }
    const profile = (byId("dummyProfileSelect") && byId("dummyProfileSelect").value) || "slow";
    state.dummyProgram.push({ type: "move_pose", pose_name: poseName, profile });
    renderDummyProgram();
    infoBox(`Added ${poseName} to Dummy Testing sequence.`);
  }

  function moveDummyProgramStep(index, delta) {
    const next = index + delta;
    if (index < 0 || next < 0 || index >= state.dummyProgram.length || next >= state.dummyProgram.length) {
      return;
    }
    const [item] = state.dummyProgram.splice(index, 1);
    state.dummyProgram.splice(next, 0, item);
    renderDummyProgram();
  }

  function removeDummyProgramStep(index) {
    if (index < 0 || index >= state.dummyProgram.length) return;
    state.dummyProgram.splice(index, 1);
    renderDummyProgram();
  }

  function applyDummyProgramToWorkspace() {
    if (!isDummyTestingActive()) {
      setTaskType("dummy_testing", { userSelected: true });
    }
    const body = state.dummyProgram.length
      ? state.dummyProgram.map((step) => ({ ...step }))
      : [{ type: "delay", ms: 500 }];
    buildWorkspaceFromFlow([
      { type: "set_task_type", task_type: "dummy_testing" },
      { type: "while_true", body },
    ]);
    infoBox("Dummy Testing sequence applied to flow.");
  }

  function taskTypeOptions() {
    const unique = [];
    (state.taskTypes || []).forEach((raw) => {
      const canonical = canonicalTaskType(raw);
      if (!unique.includes(canonical)) unique.push(canonical);
    });
    if (!unique.length) unique.push("pick_place_demo");
    return unique.map((taskType) => [taskTypeLabel(taskType), taskType]);
  }

  function normalizeFlowStepType(value) {
    const raw = String(value || "").trim().toLowerCase();
    if (
      raw === "switch_task_type" ||
      raw === "switch_task" ||
      raw === "use_task" ||
      raw === "set_task_mode" ||
      raw === "task_mode"
    ) {
      return "set_task_type";
    }
    if (raw === "whiletrue" || raw === "loop_forever") {
      return "while_true";
    }
    return raw;
  }
  function defineOperatorBlocks() {
    if (!window.Blockly || state.blocklyReady) return;

    const Blockly = window.Blockly;

    if (Blockly.common && typeof Blockly.common.defineBlocksWithJsonArray === "function") {
      Blockly.common.defineBlocksWithJsonArray([
        {
          type: "workflow_start",
          message0: "START",
          nextStatement: "FlowStep",
          colour: "#115A9B",
          tooltip: "Entry point for operator flow",
        },
        {
          type: "step_delay",
          message0: "Delay ms %1",
          args0: [
            {
              type: "field_number",
              name: "MS",
              value: 500,
              min: 0,
              max: 600000,
              precision: 10,
            },
          ],
          previousStatement: "FlowStep",
          nextStatement: "FlowStep",
          colour: "#7D8D9F",
        },
        {
          type: "step_repeat",
          message0: "Repeat count %1",
          args0: [
            {
              type: "field_number",
              name: "COUNT",
              value: 1,
              min: 1,
              max: 100000,
              precision: 1,
            },
          ],
          previousStatement: "FlowStep",
          nextStatement: "FlowStep",
          colour: "#363B44",
        },
        {
          type: "control_while_true",
          message0: "While TRUE",
          message1: "Do %1",
          args1: [
            {
              type: "input_statement",
              name: "DO",
              check: "FlowStep",
            },
          ],
          previousStatement: "FlowStep",
          nextStatement: "FlowStep",
          colour: "#363B44",
          tooltip: "Run contained blocks continuously",
        },
        {
          type: "step_switch_profile",
          message0: "Switch profile %1",
          args0: [
            {
              type: "field_dropdown",
              name: "PROFILE",
              options: ROBOT_PROFILE_OPTIONS,
            },
          ],
          previousStatement: "FlowStep",
          nextStatement: "FlowStep",
          colour: "#115A9B",
        },
        {
          type: "step_wait_for_object",
          message0: "Wait object %1 timeout s %2",
          args0: [
            {
              type: "field_input",
              name: "OBJECT_ID",
              text: "obj1",
            },
            {
              type: "field_number",
              name: "TIMEOUT_S",
              value: 2.0,
              min: 0,
              max: 600,
              precision: 0.1,
            },
          ],
          previousStatement: "FlowStep",
          nextStatement: "FlowStep",
          colour: "#12C6CF",
        },
        {
          type: "step_gripper_action",
          message0: "Gripper %1 wait ms %2",
          args0: [
            {
              type: "field_dropdown",
              name: "ACTION",
              options: [
                ["open", "open"],
                ["close", "close"],
              ],
            },
            {
              type: "field_number",
              name: "WAIT_MS",
              value: 180,
              min: 0,
              max: 10000,
              precision: 10,
            },
          ],
          previousStatement: "FlowStep",
          nextStatement: "FlowStep",
          colour: "#12C6CF",
        },
        {
          type: "step_config_vision_quality",
          message0: "Vision quality min score %1",
          args0: [
            {
              type: "field_number",
              name: "MIN_SCORE",
              value: 0.55,
              min: 0,
              max: 1,
              precision: 0.01,
            },
          ],
          message1: "match ratio %1 min inliers %2 max results %3",
          args1: [
            {
              type: "field_number",
              name: "MATCH_RATIO",
              value: 0.72,
              min: 0,
              max: 1,
              precision: 0.01,
            },
            {
              type: "field_number",
              name: "MIN_INLIERS",
              value: 8,
              min: 1,
              max: 1000,
              precision: 1,
            },
            {
              type: "field_number",
              name: "MAX_RESULTS",
              value: 1,
              min: 1,
              max: 20,
              precision: 1,
            },
          ],
          previousStatement: "FlowStep",
          nextStatement: "FlowStep",
          colour: "#AAEDF6",
        },
      ]);
    }

    Blockly.Blocks.step_capture = {
      init: function () {
        this.appendDummyInput()
          .appendField("Capture")
          .appendField(new Blockly.FieldDropdown(() => cameraOptions()), "CAMERA_ID");
        this.setPreviousStatement(true, "FlowStep");
        this.setNextStatement(true, "FlowStep");
        this.setColour("#12C6CF");
        this.setTooltip("Capture frame from selected camera");
      },
    };

    Blockly.Blocks.step_pick = {
      init: function () {
        this.appendDummyInput()
          .appendField("Visual Pick")
          .appendField(new Blockly.FieldDropdown(() => objectOptions()), "OBJECT_ID")
          .appendField("hint pose")
          .appendField(new Blockly.FieldDropdown(() => poseOptions()), "POSE_NAME");
        this.setPreviousStatement(true, "FlowStep");
        this.setNextStatement(true, "FlowStep");
        this.setColour("#115A9B");
        this.setTooltip("Detect and pick selected product");
      },
    };

    Blockly.Blocks.step_intermediate_pose = {
      init: function () {
        this.appendDummyInput()
          .appendField("Intermediate Pose")
          .appendField(new Blockly.FieldDropdown(() => poseOptions()), "POSE_NAME")
          .appendField("profile")
          .appendField(new Blockly.FieldDropdown(ROBOT_PROFILE_OPTIONS), "PROFILE");
        this.setPreviousStatement(true, "FlowStep");
        this.setNextStatement(true, "FlowStep");
        this.setColour("#0F6FB5");
      },
    };

    Blockly.Blocks.step_move_pose = {
      init: function () {
        this.appendDummyInput()
          .appendField("Move Pose")
          .appendField(new Blockly.FieldDropdown(() => poseOptions()), "POSE_NAME")
          .appendField("profile")
          .appendField(new Blockly.FieldDropdown(ROBOT_PROFILE_OPTIONS), "PROFILE");
        this.setPreviousStatement(true, "FlowStep");
        this.setNextStatement(true, "FlowStep");
        this.setColour("#1F8BCF");
      },
    };

    Blockly.Blocks.step_place = {
      init: function () {
        this.appendDummyInput()
          .appendField("Place")
          .appendField(new Blockly.FieldDropdown(() => poseOptions()), "POSE_NAME")
          .appendField("profile")
          .appendField(new Blockly.FieldDropdown(ROBOT_PROFILE_OPTIONS), "PROFILE");
        this.setPreviousStatement(true, "FlowStep");
        this.setNextStatement(true, "FlowStep");
        this.setColour("#1F8BCF");
      },
    };

    Blockly.Blocks.step_track = {
      init: function () {
        this.appendDummyInput()
          .appendField("Track")
          .appendField(new Blockly.FieldDropdown(() => objectOptions()), "OBJECT_ID")
          .appendField("for ms")
          .appendField(new Blockly.FieldNumber(1200, 0, 600000, 10), "DURATION_MS");
        this.setPreviousStatement(true, "FlowStep");
        this.setNextStatement(true, "FlowStep");
        this.setColour("#AAEDF6");
      },
    };

    Blockly.Blocks.step_set_task_type = {
      init: function () {
        this.appendDummyInput()
          .appendField("Use Task Mode")
          .appendField(new Blockly.FieldDropdown(() => taskTypeOptions()), "TASK_TYPE");
        this.setPreviousStatement(true, "FlowStep");
        this.setNextStatement(true, "FlowStep");
        this.setColour("#115A9B");
      },
    };
    Blockly.Blocks.step_config_vision_core = {
      init: function () {
        this.appendDummyInput()
          .appendField("Vision setup camera")
          .appendField(new Blockly.FieldDropdown(() => cameraOptions()), "CAMERA_ID")
          .appendField("module")
          .appendField(new Blockly.FieldDropdown(VISION_MODULE_OPTIONS), "MODULE");
        this.appendDummyInput()
          .appendField("object")
          .appendField(new Blockly.FieldDropdown(() => objectOptions()), "OBJECT_ID")
          .appendField("fps")
          .appendField(new Blockly.FieldNumber(15, 1, 120, 1), "FPS_LIMIT")
          .appendField("mode")
          .appendField(new Blockly.FieldDropdown(PROCESS_MODE_OPTIONS), "PROCESS_MODE");
        this.setPreviousStatement(true, "FlowStep");
        this.setNextStatement(true, "FlowStep");
        this.setColour("#12C6CF");
      },
    };

    Blockly.Blocks.step_config_pick_profile = {
      init: function () {
        this.appendDummyInput()
          .appendField("Pick profile approach z m")
          .appendField(new Blockly.FieldNumber(0.08, 0, 1, 0.005), "APPROACH_Z_M")
          .appendField("retreat z m")
          .appendField(new Blockly.FieldNumber(0.08, 0, 1, 0.005), "RETREAT_Z_M");
        this.appendDummyInput()
          .appendField("align with surface")
          .appendField(new Blockly.FieldCheckbox("FALSE"), "ALIGN_WITH_SURFACE");
        this.setPreviousStatement(true, "FlowStep");
        this.setNextStatement(true, "FlowStep");
        this.setColour("#115A9B");
      },
    };

    Blockly.Blocks.step_config_follow_profile = {
      init: function () {
        this.appendDummyInput()
          .appendField("Follow control")
          .appendField(
            new Blockly.FieldDropdown([
              ["velocity", "velocity"],
              ["position", "position"],
            ]),
            "CONTROL_MODE"
          )
          .appendField("mode")
          .appendField(
            new Blockly.FieldDropdown([
              ["predictive", "predictive"],
              ["direct", "direct"],
            ]),
            "FOLLOW_MODE"
          );
        this.appendDummyInput()
          .appendField("hover m")
          .appendField(new Blockly.FieldNumber(0.25, 0, 1.5, 0.01), "HOVER_HEIGHT_M")
          .appendField("rate hz")
          .appendField(new Blockly.FieldNumber(60, 1, 250, 1), "RATE_HZ");
        this.appendDummyInput()
          .appendField("max vel m/s")
          .appendField(new Blockly.FieldNumber(0.5, 0, 5, 0.01), "MAX_VEL_MPS")
          .appendField("max yaw dps")
          .appendField(new Blockly.FieldNumber(120, 1, 360, 1), "MAX_YAW_VEL_DPS");
        this.setPreviousStatement(true, "FlowStep");
        this.setNextStatement(true, "FlowStep");
        this.setColour("#115A9B");
      },
    };

    Blockly.Blocks.step_config_pallatizing_profile = {
      init: function () {
        this.appendDummyInput()
          .appendField("Pallatizing horizon s")
          .appendField(
            new Blockly.FieldNumber(2.0, 0.05, 10, 0.05),
            "PREDICTION_HORIZON_S"
          )
          .appendField("velocity scale")
          .appendField(new Blockly.FieldNumber(1.2, 0.1, 5, 0.05), "VELOCITY_SCALE");
        this.appendDummyInput()
          .appendField("dynamic pick")
          .appendField(new Blockly.FieldCheckbox("TRUE"), "DYNAMIC_PICK_ENABLED")
          .appendField("pre lead s")
          .appendField(new Blockly.FieldNumber(0.5, 0, 5, 0.05), "PRE_PICK_LEAD_S");
        this.appendDummyInput()
          .appendField("pick lead s")
          .appendField(new Blockly.FieldNumber(1.0, 0, 5, 0.05), "PICK_LEAD_S")
          .appendField("retreat lead s")
          .appendField(new Blockly.FieldNumber(1.2, 0, 5, 0.05), "RETREAT_LEAD_S");
        this.setPreviousStatement(true, "FlowStep");
        this.setNextStatement(true, "FlowStep");
        this.setColour("#363B44");
      },
    };

    state.blocklyReady = true;
  }

  function createBlocklyTheme() {
    const Blockly = window.Blockly;
    if (!Blockly || !Blockly.Theme || !Blockly.Themes) return undefined;
    return Blockly.Theme.defineTheme("dexsentOperatorTheme", {
      base: Blockly.Themes.Classic,
      componentStyles: {
        workspaceBackgroundColour: "#F6F6F6",
        toolboxBackgroundColour: "#AAEDF6",
        toolboxForegroundColour: "#363B44",
        flyoutBackgroundColour: "#FFFFFF",
        flyoutForegroundColour: "#363B44",
        flyoutOpacity: 0.95,
        scrollbarColour: "#115A9B",
        insertionMarkerColour: "#12C6CF",
        insertionMarkerOpacity: 0.35,
        cursorColour: "#115A9B",
      },
    });
  }

  function toolboxDefinition() {
    const actionBlocks = [
      { kind: "block", type: "step_capture" },
      { kind: "block", type: "step_wait_for_object" },
      { kind: "block", type: "step_pick" },
      { kind: "block", type: "step_track" },
      { kind: "block", type: "step_intermediate_pose" },
      { kind: "block", type: "step_place" },
      { kind: "block", type: "step_move_pose" },
      { kind: "block", type: "step_gripper_action" },
    ];

    const configBlocks = [
      { kind: "block", type: "step_set_task_type" },
      { kind: "block", type: "step_switch_profile" },
      { kind: "block", type: "step_config_vision_core" },
      { kind: "block", type: "step_config_vision_quality" },
      { kind: "block", type: "step_config_pick_profile" },
      { kind: "block", type: "step_config_follow_profile" },
      { kind: "block", type: "step_config_pallatizing_profile" },
    ];

    return {
      kind: "categoryToolbox",
      contents: [
        {
          kind: "category",
          name: "Task Config",
          colour: "#115A9B",
          contents: configBlocks,
        },
        {
          kind: "category",
          name: "Actions",
          colour: "#12C6CF",
          contents: actionBlocks,
        },
        {
          kind: "category",
          name: "Control",
          colour: "#363B44",
          contents: [
            { kind: "block", type: "control_while_true" },
            { kind: "block", type: "step_delay" },
            { kind: "block", type: "step_repeat" },
          ],
        },
      ],
    };
  }

  function updateToolboxForTaskType() {
    if (!state.workspace) return;
    if (typeof state.workspace.updateToolbox === "function") {
      state.workspace.updateToolbox(toolboxDefinition());
    }
  }

  function initBlocklyWorkspace() {
    if (!window.Blockly) {
      infoBox("Blockly library did not load.");
      return;
    }

    defineOperatorBlocks();

    const host = byId("blocklyWorkspace");
    if (!host) return;

    const Blockly = window.Blockly;
    state.workspace = Blockly.inject(host, {
      toolbox: toolboxDefinition(),
      renderer: "zelos",
      theme: createBlocklyTheme(),
      trashcan: true,
      sounds: false,
      move: {
        scrollbars: true,
        drag: true,
        wheel: true,
      },
      zoom: {
        controls: true,
        wheel: true,
        startScale: 0.9,
        maxScale: 1.6,
        minScale: 0.55,
        scaleSpeed: 1.16,
      },
      grid: {
        spacing: 22,
        length: 2,
        colour: "rgba(17, 90, 155, 0.08)",
        snap: true,
      },
    });

    ensureStartBlock();

    state.workspace.addChangeListener((evt) => {
      if (!evt || evt.type === Blockly.Events.UI) return;
      renderFlowPreview();
      const dragging = typeof state.workspace.isDragging === "function" && state.workspace.isDragging();
      if (isDummyTestingActive() && !dragging) {
        setDummyProgram(dummyProgramFromFlow(state.flow));
      }
    });

    state.resizeHandler = () => {
      if (!state.workspace) return;
      Blockly.svgResize(state.workspace);
    };
    window.addEventListener("resize", state.resizeHandler);
    setTimeout(() => {
      if (state.workspace) Blockly.svgResize(state.workspace);
    }, 80);
  }
  function getStartBlock() {
    if (!state.workspace) return null;
    const starts = state.workspace.getBlocksByType("workflow_start", false);
    if (!starts.length) return null;
    return starts[0];
  }

  function ensureStartBlock() {
    if (!state.workspace) return;
    const Blockly = window.Blockly;
    const starts = state.workspace.getBlocksByType("workflow_start", false);
    if (!starts.length) {
      const start = state.workspace.newBlock("workflow_start");
      start.initSvg();
      start.render();
      start.moveBy(42, 32);
    } else {
      starts.slice(1).forEach((b) => b.dispose(false));
      const one = starts[0];
      one.setMovable(false);
      one.setDeletable(false);
      const pos = one.getRelativeToSurfaceXY();
      if (pos.x < 12 || pos.y < 12) {
        one.moveBy(42 - pos.x, 32 - pos.y);
      }
    }
    renderFlowPreview();
    if (state.workspace) Blockly.svgResize(state.workspace);
  }

  function readFlowChainFromBlock(firstBlock) {
    const flow = [];
    let cursor = firstBlock;
    while (cursor) {
      const step = stepFromBlock(cursor);
      if (step) flow.push(step);
      cursor = cursor.getNextBlock();
    }
    return flow;
  }

  function stepFromBlock(block) {
    if (!block) return null;
    switch (block.type) {
      case "control_while_true":
        return {
          type: "while_true",
          body: readFlowChainFromBlock(block.getInputTargetBlock("DO")),
        };
      case "step_set_task_type":
        return {
          type: "set_task_type",
          task_type: canonicalTaskType(block.getFieldValue("TASK_TYPE") || "pick_place_demo"),
        };
      case "step_capture":
        return {
          type: "capture",
          camera_id: block.getFieldValue("CAMERA_ID") || "",
          note: "capture frame",
        };
      case "step_pick":
        return {
          type: "pick",
          object_id: block.getFieldValue("OBJECT_ID") || "",
          pose_name: block.getFieldValue("POSE_NAME") || "",
        };
      case "step_intermediate_pose":
        return {
          type: "intermediate_pose",
          pose_name: block.getFieldValue("POSE_NAME") || "",
          profile: block.getFieldValue("PROFILE") || "normal",
        };
      case "step_move_pose":
        return {
          type: "move_pose",
          pose_name: block.getFieldValue("POSE_NAME") || "",
          profile: block.getFieldValue("PROFILE") || "normal",
        };
      case "step_place":
        return {
          type: "place",
          pose_name: block.getFieldValue("POSE_NAME") || "",
          profile: block.getFieldValue("PROFILE") || "normal",
        };
      case "step_delay":
        return {
          type: "delay",
          ms: toSafeNumber(block.getFieldValue("MS"), 500, 0),
        };
      case "step_track":
        return {
          type: "track",
          object_id: block.getFieldValue("OBJECT_ID") || "",
          duration_ms: toSafeNumber(block.getFieldValue("DURATION_MS"), 1200, 0),
        };
      case "step_repeat":
        return {
          type: "repeat",
          count: toSafeInteger(block.getFieldValue("COUNT"), 1, 1),
        };
      case "step_wait_for_object":
        return {
          type: "wait_for_object",
          object_id: block.getFieldValue("OBJECT_ID") || "",
          timeout_s: toSafeNumber(block.getFieldValue("TIMEOUT_S"), 2.0, 0),
        };
      case "step_gripper_action":
        return {
          type: "gripper_action",
          action: block.getFieldValue("ACTION") || "close",
          wait_ms: toSafeInteger(block.getFieldValue("WAIT_MS"), 180, 0),
        };
      case "step_switch_profile":
        return {
          type: "switch_profile",
          profile: block.getFieldValue("PROFILE") || "normal",
        };
      case "step_config_vision_core":
        return {
          type: "config_vision_core",
          camera_id: block.getFieldValue("CAMERA_ID") || "",
          module: block.getFieldValue("MODULE") || "tamplate_matching_sift",
          object_id: block.getFieldValue("OBJECT_ID") || "",
          fps_limit: toSafeInteger(block.getFieldValue("FPS_LIMIT"), 15, 1),
          process_mode: block.getFieldValue("PROCESS_MODE") || "continuous",
        };
      case "step_config_vision_quality":
        return {
          type: "config_vision_quality",
          min_score: toSafeNumber(block.getFieldValue("MIN_SCORE"), 0.55, 0),
          match_ratio: toSafeNumber(block.getFieldValue("MATCH_RATIO"), 0.72, 0),
          min_inliers: toSafeInteger(block.getFieldValue("MIN_INLIERS"), 8, 1),
          max_results: toSafeInteger(block.getFieldValue("MAX_RESULTS"), 1, 1),
        };
      case "step_config_pick_profile":
        return {
          type: "config_pick_profile",
          approach_z_m: toSafeNumber(block.getFieldValue("APPROACH_Z_M"), 0.08, 0),
          retreat_z_m: toSafeNumber(block.getFieldValue("RETREAT_Z_M"), 0.08, 0),
          align_with_surface: fieldToBool(block, "ALIGN_WITH_SURFACE"),
        };
      case "step_config_follow_profile":
        return {
          type: "config_follow_profile",
          control_mode: block.getFieldValue("CONTROL_MODE") || "velocity",
          follow_mode: block.getFieldValue("FOLLOW_MODE") || "predictive",
          hover_height_m: toSafeNumber(block.getFieldValue("HOVER_HEIGHT_M"), 0.25, 0),
          rate_hz: toSafeInteger(block.getFieldValue("RATE_HZ"), 60, 1),
          max_vel_mps: toSafeNumber(block.getFieldValue("MAX_VEL_MPS"), 0.5, 0),
          max_yaw_vel_dps: toSafeNumber(block.getFieldValue("MAX_YAW_VEL_DPS"), 120, 1),
        };
      case "step_config_pallatizing_profile":
        return {
          type: "config_pallatizing_profile",
          prediction_horizon_s: toSafeNumber(
            block.getFieldValue("PREDICTION_HORIZON_S"),
            2.0,
            0
          ),
          velocity_scale: toSafeNumber(block.getFieldValue("VELOCITY_SCALE"), 1.2, 0.1),
          dynamic_pick_enabled: fieldToBool(block, "DYNAMIC_PICK_ENABLED"),
          pre_pick_lead_s: toSafeNumber(block.getFieldValue("PRE_PICK_LEAD_S"), 0.5, 0),
          pick_lead_s: toSafeNumber(block.getFieldValue("PICK_LEAD_S"), 1.0, 0),
          retreat_lead_s: toSafeNumber(block.getFieldValue("RETREAT_LEAD_S"), 1.2, 0),
        };
      default:
        return null;
    }
  }

  function blockTypeFromStep(stepType) {
    const map = {
      while_true: "control_while_true",
      set_task_type: "step_set_task_type",
      capture: "step_capture",
      pick: "step_pick",
      intermediate_pose: "step_intermediate_pose",
      move_pose: "step_move_pose",
      place: "step_place",
      delay: "step_delay",
      track: "step_track",
      repeat: "step_repeat",
      wait_for_object: "step_wait_for_object",
      gripper_action: "step_gripper_action",
      switch_profile: "step_switch_profile",
      config_vision_core: "step_config_vision_core",
      config_vision_quality: "step_config_vision_quality",
      config_pick_profile: "step_config_pick_profile",
      config_follow_profile: "step_config_follow_profile",
      config_pallatizing_profile: "step_config_pallatizing_profile",
    };
    return map[normalizeFlowStepType(stepType)] || "";
  }

  function applyStepToBlock(block, step) {
    if (!block || !step) return;
    const type = block.type;
    if (type === "step_set_task_type") {
      safeSetField(
        block,
        "TASK_TYPE",
        canonicalTaskType(step.task_type || step.mode || "pick_place_demo")
      );
    } else if (type === "step_capture") {
      safeSetField(block, "CAMERA_ID", step.camera_id || "");
    } else if (type === "step_pick") {
      safeSetField(block, "OBJECT_ID", step.object_id || "");
      safeSetField(block, "POSE_NAME", step.pose_name || "");
    } else if (type === "step_intermediate_pose") {
      safeSetField(block, "POSE_NAME", step.pose_name || "");
      safeSetField(block, "PROFILE", step.profile || "normal");
    } else if (type === "step_move_pose") {
      safeSetField(block, "POSE_NAME", step.pose_name || "");
      safeSetField(block, "PROFILE", step.profile || "normal");
    } else if (type === "step_place") {
      safeSetField(block, "POSE_NAME", step.pose_name || "");
      safeSetField(block, "PROFILE", step.profile || "normal");
    } else if (type === "step_delay") {
      safeSetField(block, "MS", toSafeNumber(step.ms, 500, 0));
    } else if (type === "step_track") {
      safeSetField(block, "OBJECT_ID", step.object_id || "");
      safeSetField(block, "DURATION_MS", toSafeNumber(step.duration_ms, 1200, 0));
    } else if (type === "step_repeat") {
      safeSetField(block, "COUNT", toSafeInteger(step.count, 1, 1));
    } else if (type === "step_wait_for_object") {
      safeSetField(block, "OBJECT_ID", step.object_id || "");
      safeSetField(block, "TIMEOUT_S", toSafeNumber(step.timeout_s, 2, 0));
    } else if (type === "step_gripper_action") {
      safeSetField(block, "ACTION", step.action || "close");
      safeSetField(block, "WAIT_MS", toSafeInteger(step.wait_ms, 180, 0));
    } else if (type === "step_switch_profile") {
      safeSetField(block, "PROFILE", step.profile || "normal");
    } else if (type === "step_config_vision_core") {
      safeSetField(block, "CAMERA_ID", step.camera_id || "");
      safeSetField(block, "MODULE", step.module || "tamplate_matching_sift");
      safeSetField(block, "OBJECT_ID", step.object_id || "");
      safeSetField(block, "FPS_LIMIT", toSafeInteger(step.fps_limit, 15, 1));
      safeSetField(block, "PROCESS_MODE", step.process_mode || "continuous");
    } else if (type === "step_config_vision_quality") {
      safeSetField(block, "MIN_SCORE", toSafeNumber(step.min_score, 0.55, 0));
      safeSetField(block, "MATCH_RATIO", toSafeNumber(step.match_ratio, 0.72, 0));
      safeSetField(block, "MIN_INLIERS", toSafeInteger(step.min_inliers, 8, 1));
      safeSetField(block, "MAX_RESULTS", toSafeInteger(step.max_results, 1, 1));
    } else if (type === "step_config_pick_profile") {
      safeSetField(block, "APPROACH_Z_M", toSafeNumber(step.approach_z_m, 0.08, 0));
      safeSetField(block, "RETREAT_Z_M", toSafeNumber(step.retreat_z_m, 0.08, 0));
      safeSetField(
        block,
        "ALIGN_WITH_SURFACE",
        boolToField(step.align_with_surface === true)
      );
    } else if (type === "step_config_follow_profile") {
      safeSetField(block, "CONTROL_MODE", step.control_mode || "velocity");
      safeSetField(block, "FOLLOW_MODE", step.follow_mode || "predictive");
      safeSetField(block, "HOVER_HEIGHT_M", toSafeNumber(step.hover_height_m, 0.25, 0));
      safeSetField(block, "RATE_HZ", toSafeInteger(step.rate_hz, 60, 1));
      safeSetField(block, "MAX_VEL_MPS", toSafeNumber(step.max_vel_mps, 0.5, 0));
      safeSetField(
        block,
        "MAX_YAW_VEL_DPS",
        toSafeNumber(step.max_yaw_vel_dps, 120, 1)
      );
    } else if (type === "step_config_pallatizing_profile") {
      safeSetField(
        block,
        "PREDICTION_HORIZON_S",
        toSafeNumber(step.prediction_horizon_s, 2.0, 0)
      );
      safeSetField(block, "VELOCITY_SCALE", toSafeNumber(step.velocity_scale, 1.2, 0.1));
      safeSetField(
        block,
        "DYNAMIC_PICK_ENABLED",
        boolToField(step.dynamic_pick_enabled !== false)
      );
      safeSetField(block, "PRE_PICK_LEAD_S", toSafeNumber(step.pre_pick_lead_s, 0.5, 0));
      safeSetField(block, "PICK_LEAD_S", toSafeNumber(step.pick_lead_s, 1.0, 0));
      safeSetField(block, "RETREAT_LEAD_S", toSafeNumber(step.retreat_lead_s, 1.2, 0));
    }
  }
  function collectConnectedFlowBlockIds(firstBlock, ids) {
    let cursor = firstBlock;
    while (cursor) {
      if (ACTION_BLOCK_TYPES.has(cursor.type)) {
        ids.add(cursor.id);
      }
      if (cursor.type === "control_while_true") {
        collectConnectedFlowBlockIds(cursor.getInputTargetBlock("DO"), ids);
      }
      cursor = cursor.getNextBlock();
    }
  }

  function readFlowFromWorkspace() {
    if (!state.workspace) return { flow: [], orphanCount: 0 };

    const start = getStartBlock();
    if (!start) return { flow: [], orphanCount: 0 };

    const flow = readFlowChainFromBlock(start.getNextBlock());
    const connectedBlockIds = new Set();
    collectConnectedFlowBlockIds(start.getNextBlock(), connectedBlockIds);

    const totalActionBlocks = state.workspace
      .getAllBlocks(false)
      .filter((b) => ACTION_BLOCK_TYPES.has(b.type)).length;
    const orphanCount = Math.max(0, totalActionBlocks - connectedBlockIds.size);
    return { flow, orphanCount };
  }

  function buildFlowChainBlocks(flow, parentConnection, startX, startY) {
    let previous = null;
    let skipped = 0;
    let visualIndex = 0;

    (flow || []).forEach((rawStep) => {
      const step = rawStep && typeof rawStep === "object" ? rawStep : {};
      const blockType = blockTypeFromStep(step.type);
      if (!blockType) {
        skipped += 1;
        return;
      }

      const block = state.workspace.newBlock(blockType);
      applyStepToBlock(block, step);
      block.initSvg();
      block.render();
      block.moveBy(startX, startY + visualIndex * 78);

      if (previous && previous.nextConnection && block.previousConnection) {
        previous.nextConnection.connect(block.previousConnection);
      } else if (parentConnection && block.previousConnection) {
        parentConnection.connect(block.previousConnection);
      }

      if (block.type === "control_while_true") {
        const bodyFlow = Array.isArray(step.body) ? step.body : [];
        const doInput = block.getInput("DO");
        const doConnection = doInput && doInput.connection;
        if (doConnection && bodyFlow.length) {
          const bodyResult = buildFlowChainBlocks(
            bodyFlow,
            doConnection,
            startX + 56,
            startY + visualIndex * 78 + 62
          );
          skipped += bodyResult.skipped;
        }
      }

      previous = block;
      visualIndex += 1;
    });

    return { skipped };
  }

  function buildWorkspaceFromFlow(flow) {
    if (!state.workspace) return;

    state.workspace.clear();

    const start = state.workspace.newBlock("workflow_start");
    start.initSvg();
    start.render();
    start.moveBy(42, 32);
    start.setMovable(false);
    start.setDeletable(false);

    const built = buildFlowChainBlocks(flow || [], start.nextConnection, 72, 108);

    if (built.skipped > 0) {
      infoBox(`Loaded flow with ${built.skipped} unsupported step(s) skipped.`);
    }

    normalizeFlowLayout();
    renderFlowPreview();
  }

  function summarizeStep(step) {
    if (!step) return "-";
    const t = normalizeFlowStepType(step.type);
    if (t === "while_true") return "While TRUE";
    if (t === "set_task_type") {
      return `Use task mode ${taskTypeLabel(step.task_type || "pick_place_demo")}`;
    }
    if (t === "config_vision_core") {
      return `Vision ${step.module || "-"} | ${step.object_id || "-"} @ ${step.fps_limit || "-"} fps`;
    }
    if (t === "config_vision_quality") {
      return `Vision quality score ${step.min_score || "-"} ratio ${step.match_ratio || "-"}`;
    }
    if (t === "config_pick_profile") {
      return `Pick profile z(${step.approach_z_m || "-"}/${step.retreat_z_m || "-"})`; 
    }
    if (t === "config_follow_profile") {
      return `Follow ${step.control_mode || "-"} ${step.follow_mode || "-"} @ ${step.rate_hz || "-"} hz`;
    }
    if (t === "config_pallatizing_profile") {
      return `Pallatizing horizon ${step.prediction_horizon_s || "-"}s scale ${step.velocity_scale || "-"}`;
    }
    if (t === "switch_profile") return `Switch profile ${step.profile || "normal"}`;
    if (t === "capture") return `Capture (${step.camera_id || "any"})`;
    if (t === "wait_for_object") return `Wait ${step.object_id || "object"} ${step.timeout_s || "-"}s`;
    if (t === "pick") return `Visual Pick ${step.object_id || "-"}`;
    if (t === "track") return `Track ${step.object_id || "-"} for ${toSafeNumber(step.duration_ms, 0, 0)} ms`;
    if (t === "intermediate_pose") return `Intermediate ${step.pose_name || "-"}`;
    if (t === "move_pose") return `Move ${step.pose_name || "-"}`;
    if (t === "place") return `Place ${step.pose_name || "-"}`;
    if (t === "gripper_action") return `Gripper ${step.action || "-"}`;
    if (t === "delay") return `Delay ${toSafeNumber(step.ms, 0, 0)} ms`;
    if (t === "repeat") return `Repeat ${toSafeNumber(step.count, 1, 1)} cycle(s)`;
    return String(step.type || "unknown");
  }

  function appendPreviewLines(flow, path, depth, lines) {
    (flow || []).forEach((step, idx) => {
      const stepPath = (path || []).concat(idx + 1);
      const prefix = `${stepPath.join(".")}. `;
      const pad = "  ".repeat(depth || 0);
      const stepType = normalizeFlowStepType(step && step.type);

      if (stepType === "while_true") {
        lines.push(`${pad}${prefix}While TRUE`);
        const body = Array.isArray(step && step.body) ? step.body : [];
        if (!body.length) {
          lines.push(`${pad}  (empty loop body)`);
        } else {
          appendPreviewLines(body, stepPath, (depth || 0) + 1, lines);
        }
        return;
      }

      lines.push(`${pad}${prefix}${summarizeStep(step)}`);
    });
  }

  function renderFlowPreview() {
    const out = readFlowFromWorkspace();
    state.flow = out.flow;

    const preview = byId("flowPreview");
    if (!preview) return;

    if (!out.flow.length) {
      preview.textContent = "No steps yet. Drag blocks from toolbox and connect below START.";
      return;
    }

    const lines = [];
    appendPreviewLines(out.flow, [], 0, lines);
    if (out.orphanCount > 0) {
      lines.push("");
      lines.push(`Ignored (not connected to START): ${out.orphanCount} block(s)`);
    }
    preview.textContent = lines.join("\n");
  }

  function layoutFlowChain(firstBlock, startX, startY) {
    let cursor = firstBlock;
    let currentY = startY;
    while (cursor) {
      const pos = cursor.getRelativeToSurfaceXY();
      cursor.moveBy(startX - pos.x, currentY - pos.y);

      let nextY = currentY + 78;
      if (cursor.type === "control_while_true") {
        const bodyStart = cursor.getInputTargetBlock("DO");
        if (bodyStart) {
          const bodyBottom = layoutFlowChain(bodyStart, startX + 56, currentY + 62);
          nextY = Math.max(nextY, bodyBottom + 24);
        } else {
          nextY = Math.max(nextY, currentY + 94);
        }
      }

      currentY = nextY;
      cursor = cursor.getNextBlock();
    }
    return currentY;
  }

  function normalizeFlowLayout() {
    if (!state.workspace) return;
    const start = getStartBlock();
    if (!start) return;

    const baseX = 42;
    const baseY = 32;
    const startPos = start.getRelativeToSurfaceXY();
    start.moveBy(baseX - startPos.x, baseY - startPos.y);

    layoutFlowChain(start.getNextBlock(), baseX + 24, baseY + 84);
  }

  function inferFlowFromTaskPayload(taskType, payload) {
    const task = payload && typeof payload === "object" ? payload : {};
    const vision = task.vision && typeof task.vision === "object" ? task.vision : {};
    const robot = task.robot && typeof task.robot === "object" ? task.robot : {};
    const canonicalType = canonicalTaskType(taskType);
    const flow = [];
    const loopBody = [];

    const inferredObject = String(
      vision.object_id ||
        (vision.params && vision.params.object_id) ||
        ""
    ).trim();

    flow.push({
      type: "set_task_type",
      task_type: canonicalType,
    });

    if (Object.keys(vision).length) {
      flow.push({
        type: "config_vision_core",
        camera_id: vision.camera_id || "",
        module: vision.module || "tamplate_matching_sift",
        object_id: inferredObject,
        fps_limit: toSafeInteger(vision.fps_limit, 15, 1),
        process_mode: vision.process_mode || "continuous",
      });

      const params = vision.params && typeof vision.params === "object" ? vision.params : {};
      const filters = vision.filters && typeof vision.filters === "object" ? vision.filters : {};
      flow.push({
        type: "config_vision_quality",
        min_score: toSafeNumber(
          params.min_score !== undefined ? params.min_score : filters.min_score,
          0.55,
          0
        ),
        match_ratio: toSafeNumber(params.match_ratio, 0.72, 0),
        min_inliers: toSafeInteger(
          params.min_inliers !== undefined ? params.min_inliers : filters.min_inliers,
          8,
          1
        ),
        max_results: toSafeInteger(params.max_results, 1, 1),
      });
    }

    if (robot.default_profile) {
      flow.push({ type: "switch_profile", profile: String(robot.default_profile) });
    }

    if (canonicalType === "follow_object") {
      const follow = robot.follow && typeof robot.follow === "object" ? robot.follow : {};
      flow.push({
        type: "config_follow_profile",
        control_mode: follow.control_mode || "velocity",
        follow_mode: follow.mode || "predictive",
        hover_height_m: toSafeNumber(follow.hover_height_m, 0.25, 0),
        rate_hz: toSafeInteger(follow.rate_hz, 60, 1),
        max_vel_mps: toSafeNumber(follow.max_vel_mps, 0.5, 0),
        max_yaw_vel_dps: toSafeNumber(follow.max_yaw_vel_dps, 120, 1),
      });
      loopBody.push({ type: "capture", camera_id: vision.camera_id || "" });
      loopBody.push({
        type: "wait_for_object",
        object_id: inferredObject,
        timeout_s: toSafeNumber(follow.no_match_timeout_s, 2.0, 0),
      });
      loopBody.push({
        type: "track",
        object_id: inferredObject,
        duration_ms: toSafeInteger((follow.max_target_age_s || 1) * 1000, 1200, 0),
      });
      flow.push({ type: "while_true", body: loopBody });
      return flow;
    }

    if (canonicalType === "dummy_testing") {
      const program = Array.isArray(task.dummy_testing?.program) ? task.dummy_testing.program : [];
      program.forEach((step) => {
        if (step && typeof step === "object") loopBody.push(step);
      });
      if (!loopBody.length) {
        const poseNames = state.poses.map((p) => p.name).filter(Boolean).slice(0, 3);
        poseNames.forEach((poseName) => {
          loopBody.push({ type: "move_pose", pose_name: poseName, profile: robot.default_profile || "slow" });
        });
      }
      if (!loopBody.length) loopBody.push({ type: "delay", ms: 500 });
      flow.push({ type: "while_true", body: loopBody });
      return flow;
    }

    const pick = robot.pick && typeof robot.pick === "object" ? robot.pick : {};
    flow.push({
      type: "config_pick_profile",
      approach_z_m: toSafeNumber(
        Array.isArray(pick.approach_offset_m) ? pick.approach_offset_m[2] : undefined,
        0.08,
        0
      ),
      retreat_z_m: toSafeNumber(
        Array.isArray(pick.retreat_offset_m) ? pick.retreat_offset_m[2] : undefined,
        0.08,
        0
      ),
      align_with_surface: !!pick.align_with_surface,
    });

    if (canonicalType === "pallatizing") {
      const pall = task.pallatizing && typeof task.pallatizing === "object" ? task.pallatizing : {};
      flow.push({
        type: "config_pallatizing_profile",
        prediction_horizon_s: toSafeNumber(pall.prediction_horizon_s, 2.0, 0),
        velocity_scale: toSafeNumber(pall.velocity_scale, 1.2, 0.1),
        dynamic_pick_enabled: pall.dynamic_pick_enabled !== false,
        pre_pick_lead_s: toSafeNumber(pall.pre_pick_lead_s, 0.5, 0),
        pick_lead_s: toSafeNumber(pall.pick_lead_s, 1.0, 0),
        retreat_lead_s: toSafeNumber(pall.retreat_lead_s, 1.2, 0),
      });
    }

    const capturePoseName = String(
      robot.capture_pose_name ||
        (robot.capture_pose && robot.capture_pose.name) ||
        ""
    ).trim();
    if (capturePoseName) {
      loopBody.push({
        type: "move_pose",
        pose_name: capturePoseName,
        profile: robot.default_profile || "slow",
      });
    } else {
      loopBody.push({ type: "capture", camera_id: vision.camera_id || "" });
    }
    const intermediatePoseName = String(
      robot.intermediate_pose_name ||
        robot.pick_intermediate_pose_name ||
        (robot.intermediate_pose && robot.intermediate_pose.name) ||
        ""
    ).trim();
    if (intermediatePoseName) {
      loopBody.push({
        type: "intermediate_pose",
        pose_name: intermediatePoseName,
        profile: robot.intermediate_profile || robot.default_profile || "slow",
      });
    }

    loopBody.push({
      type: "wait_for_object",
      object_id: inferredObject,
      timeout_s: toSafeNumber(vision.timeout_s, 2.0, 0),
    });
    loopBody.push({ type: "pick", object_id: inferredObject, pose_name: "" });
    const placePoseName = String(
      robot.place_pose_name ||
        (robot.place_pose && robot.place_pose.name) ||
        ""
    ).trim();
    if (placePoseName) {
      loopBody.push({
        type: "place",
        pose_name: placePoseName,
        profile: (robot.place && robot.place.profile) || "normal",
      });
    }
    if (!loopBody.length) {
      loopBody.push({ type: "delay", ms: 500 });
    }

    flow.push({ type: "while_true", body: loopBody });

    return flow;
  }
  async function refreshStations() {
    const res = await window.operatorApi("/stations", {}, { silent: true });
    if (!res.ok || !res.body) {
      infoBox("Cannot reach station list.");
      return;
    }
    state.stations = res.body.stations || [];
    fillSelect(
      "stationSelect",
      state.stations,
      (s) => s.station_id,
      (s) => s.name || s.station_id,
      state.currentStationId
    );
    const stationSelect = byId("stationSelect");
    state.currentStationId = (stationSelect && stationSelect.value) || "";
  }

  async function refreshTaskTypes() {
    const res = await window.operatorApi("/task_types", {}, { silent: true });
    const discovered = [];
    if (res.ok && res.body && Array.isArray(res.body.task_types)) {
      (res.body.task_types || []).forEach((raw) => {
        const canonical = canonicalTaskType(raw);
        if (!discovered.includes(canonical)) discovered.push(canonical);
      });
    }
    BASE_TASK_TYPES.forEach((base) => {
      if (!discovered.includes(base)) discovered.push(base);
    });
    state.taskTypes = discovered.length ? discovered : [...BASE_TASK_TYPES];
    if (!state.taskTypes.includes(state.currentTaskType)) {
      state.currentTaskType = state.taskTypes[0] || "pick_place_demo";
    }
    refreshTaskTypeSelect();
  }

  async function refreshAssets() {
    if (!state.currentStationId) {
      state.assets = [];
      fillSelect("assetSelect", [], () => "", () => "", "");
      return;
    }
    const res = await window.operatorApi(
      `/stations/${encodeURIComponent(state.currentStationId)}/processes`,
      {},
      { silent: true }
    );
    if (!res.ok || !res.body) {
      infoBox("Failed loading assets.");
      return;
    }
    state.assets = res.body.processes || [];
    fillSelect(
      "assetSelect",
      state.assets,
      (p) => assetId(p),
      (p) => p.name || assetId(p),
      state.currentAssetId
    );
    const assetSelect = byId("assetSelect");
    state.currentAssetId = (assetSelect && assetSelect.value) || "";
    const active = selectedAsset();
    byId("assetName").value = (active && active.name) || "";
  }

  async function refreshTasks() {
    if (!state.currentAssetId) {
      state.tasks = [];
      fillSelect("taskSelect", [], () => "", () => "", "");
      return;
    }
    const res = await window.operatorApi(
      `/processes/${encodeURIComponent(state.currentAssetId)}/tasks`,
      {},
      { silent: true }
    );
    if (!res.ok || !res.body) {
      infoBox("Failed loading tasks.");
      return;
    }
    state.tasks = res.body.tasks || [];
    fillSelect(
      "taskSelect",
      state.tasks,
      (t) => t.task_id,
      (t) => t.name || t.task_id,
      state.currentTaskId
    );
    const taskSelect = byId("taskSelect");
    state.currentTaskId = (taskSelect && taskSelect.value) || "";

    const task = selectedTask() || state.tasks[0] || null;
    if (task) {
      setTaskType(task.task_type || state.currentTaskType);
    }
  }

  async function refreshPoses() {
    if (!state.currentAssetId) {
      state.poses = [];
      fillSelect("poseSelect", [], () => "", () => "", "");
      return;
    }
    const res = await window.operatorApi(
      `/processes/${encodeURIComponent(state.currentAssetId)}/poses`,
      {},
      { silent: true }
    );
    if (!res.ok || !res.body) {
      infoBox("Failed loading poses.");
      return;
    }
    state.poses = res.body.poses || [];
    fillSelect(
      "poseSelect",
      state.poses,
      (p) => p.name,
      (p) => p.name,
      ((byId("poseSelect") && byId("poseSelect").value) || "")
    );
    byId("poseInfo").textContent = `poses: ${state.poses.length}`;
    renderTaskParamsPanel();
    renderFlowPreview();
  }

  async function refreshObjects() {
    if (!state.currentAssetId) {
      state.objects = [];
      renderObjects();
      return;
    }
    const res = await window.operatorApi(
      `/processes/${encodeURIComponent(state.currentAssetId)}/objects`,
      {},
      { silent: true }
    );
    if (!res.ok || !res.body) {
      infoBox("Failed loading products.");
      return;
    }
    state.objects = res.body.objects || [];
    renderObjects();
    renderFlowPreview();
  }

  function renderObjects() {
    const host = byId("objectList");
    if (!host) return;
    if (!state.objects.length) {
      host.innerHTML = '<div class="hint">No product templates yet.</div>';
      return;
    }
    host.innerHTML = state.objects
      .map((obj) => {
        const templates = Array.isArray(obj.templates) ? obj.templates.length : 0;
        return `<div class="object-item"><span>${sanitize(obj.object_id || "")}</span><span>${templates} templates</span></div>`;
      })
      .join("");
  }

  function parseTaskPayloadFromResponse(body) {
    if (!body || typeof body !== "object") return {};
    if (body.task && typeof body.task === "object") return body.task;
    return {};
  }

  function deepCloneFlow(flow) {
    try {
      return JSON.parse(JSON.stringify(Array.isArray(flow) ? flow : []));
    } catch (_) {
      return [];
    }
  }

  function collectTaskTypesFromFlow(flow, out) {
    (flow || []).forEach((step) => {
      const stepType = normalizeFlowStepType(step && step.type);
      if (stepType === "set_task_type") {
        const canonical = canonicalTaskType(step.task_type || "pick_place_demo");
        if (!out.includes(canonical)) out.push(canonical);
      }
      if (stepType === "while_true" && Array.isArray(step && step.body)) {
        collectTaskTypesFromFlow(step.body, out);
      }
    });
  }

  function taskTypesUsedInFlow(flow) {
    const taskTypes = [];
    collectTaskTypesFromFlow(flow || [], taskTypes);
    if (!taskTypes.includes(state.currentTaskType)) {
      taskTypes.unshift(state.currentTaskType);
    }
    return taskTypes;
  }

  function firstStepOfType(flow, wantedType) {
    let found = null;
    function visit(steps) {
      (steps || []).forEach((step) => {
        if (found || !step || typeof step !== "object") return;
        const stepType = normalizeFlowStepType(step.type);
        if (stepType === wantedType) {
          found = step;
          return;
        }
        if (Array.isArray(step.body)) visit(step.body);
      });
    }
    visit(flow || []);
    return found;
  }

  async function loadTaskFlow() {
    if (!state.currentTaskId) {
      state.currentTaskPayload = {};
      setDummyProgram([]);
      renderTaskParamsPanel();
      buildWorkspaceFromFlow([]);
      return;
    }

    const res = await window.operatorApi(
      `/tasks/${encodeURIComponent(state.currentTaskId)}`,
      {},
      { silent: true }
    );

    if (!res.ok || !res.body) {
      infoBox("Task flow load failed.");
      return;
    }

    const taskDoc = res.body;
    const taskPayload = parseTaskPayloadFromResponse(taskDoc);
    state.currentTaskPayload = taskPayload;
    const current = selectedTask();
    const taskType =
      taskDoc.task_type || (current && current.task_type) || state.currentTaskType;
    setTaskType(taskType);

    let savedFlow = Array.isArray(taskPayload.operator_flow) ? taskPayload.operator_flow : [];
    if (!savedFlow.length) {
      savedFlow = inferFlowFromTaskPayload(state.currentTaskType, taskPayload);
      if (savedFlow.length) {
        infoBox("Loaded generated flow from current task configuration.");
      }
    }

    buildWorkspaceFromFlow(deepCloneFlow(savedFlow));
    const dummyCfg =
      taskPayload.dummy_testing && typeof taskPayload.dummy_testing === "object"
        ? taskPayload.dummy_testing
        : {};
    const dummyProgram = Array.isArray(dummyCfg.program)
      ? dummyCfg.program
      : dummyProgramFromFlow(savedFlow);
    setDummyProgram(dummyProgram);
    const loopInput = byId("dummyLoopIterations");
    if (loopInput) loopInput.value = toSafeInteger(dummyCfg.max_loop_iterations, 1, 1);
    renderTaskParamsPanel();
  }
  async function saveFlow() {
    if (!state.currentTaskId) {
      infoBox("Select a task first.");
      return;
    }

    const parsed = readFlowFromWorkspace();
    let flow = parsed.flow;
    const dummyActive = isDummyTestingActive();
    const dummyProgram = dummyActive
      ? (state.dummyProgram.length ? state.dummyProgram.map((step) => ({ ...step })) : dummyProgramFromFlow(flow))
      : [];
    if (dummyActive && dummyProgram.length) {
      flow = [
        { type: "set_task_type", task_type: "dummy_testing" },
        { type: "while_true", body: dummyProgram },
      ];
    }
    const flowTaskTypes = taskTypesUsedInFlow(flow);
    const taskPayload = {
      operator_flow: flow,
      operator_flow_version: 2,
      operator_flow_task_types: flowTaskTypes,
      ui_mode: "operator",
    };
    const intermediateStep = firstStepOfType(flow, "intermediate_pose");
    if (intermediateStep) {
      const poseName = String(intermediateStep.pose_name || "").trim();
      taskPayload.robot = {
        intermediate_pose_name: poseName,
        intermediate_pose: poseName
          ? { name: poseName, pose_path: `poses/${poseName}.json` }
          : {},
        intermediate_profile: intermediateStep.profile || "slow",
        intermediate_move_strategy: "cartesian",
      };
    } else {
      taskPayload.robot = {
        intermediate_pose_name: "",
        intermediate_pose: {},
      };
    }
    if (dummyActive) {
      const currentCfg =
        state.currentTaskPayload &&
        state.currentTaskPayload.dummy_testing &&
        typeof state.currentTaskPayload.dummy_testing === "object"
          ? state.currentTaskPayload.dummy_testing
          : {};
      taskPayload.dummy_testing = {
        ...currentCfg,
        program: dummyProgram,
        profile: (byId("dummyProfileSelect") && byId("dummyProfileSelect").value) || currentCfg.profile || "slow",
        max_loop_iterations: toSafeInteger(
          byId("dummyLoopIterations") && byId("dummyLoopIterations").value,
          currentCfg.max_loop_iterations || 1,
          1
        ),
      };
    }
    const payload = {
      task_type: flowTaskTypes[0] || state.currentTaskType,
      task: taskPayload,
    };

    const res = await window.operatorApi(
      `/tasks/${encodeURIComponent(state.currentTaskId)}`,
      {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }
    );

    if (!res.ok) {
      infoBox("Flow save failed.");
      return;
    }

    if (parsed.orphanCount > 0) {
      infoBox(`Flow saved. ${parsed.orphanCount} disconnected block(s) ignored.`);
      return;
    }

    infoBox("Flow saved.");
    state.currentTaskPayload = taskPayload;
    await refreshTasks();
  }

  async function saveAssetName() {
    if (!state.currentAssetId) return;
    const name = byId("assetName").value.trim();
    const res = await window.operatorApi(
      `/processes/${encodeURIComponent(state.currentAssetId)}`,
      {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      }
    );
    if (!res.ok) {
      infoBox("Asset save failed.");
      return;
    }
    await refreshAssets();
    infoBox("Asset updated.");
  }

  async function createTask() {
    if (!state.currentAssetId) {
      infoBox("Select an asset first.");
      return;
    }
    const name = byId("newTaskName").value.trim() || defaultTaskNameForType(state.currentTaskType);
    const taskBody = {
      ui_mode: "operator",
      operator_flow_version: 2,
      operator_flow: [],
      operator_flow_task_types: [state.currentTaskType],
    };
    if (isDummyTestingActive()) {
      taskBody.dummy_testing = {
        program: [],
        profile: (byId("dummyProfileSelect") && byId("dummyProfileSelect").value) || "slow",
        strict_collision: true,
        dry_run: false,
        max_loop_iterations: toSafeInteger(
          byId("dummyLoopIterations") && byId("dummyLoopIterations").value,
          1,
          1
        ),
      };
    }
    const res = await window.operatorApi(
      `/processes/${encodeURIComponent(state.currentAssetId)}/tasks`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name,
          task_type: state.currentTaskType,
          task: taskBody,
        }),
      }
    );
    if (!res.ok || !res.body) {
      infoBox("Task create failed.");
      return;
    }
    state.currentTaskId = res.body.task_id || "";
    byId("newTaskName").value = "";
    await refreshTasks();
    await loadTaskFlow();
    infoBox("Task created.");
  }

  async function recordPose() {
    if (!state.currentAssetId) {
      infoBox("Select an asset first.");
      return;
    }
    const name = byId("poseNameInput").value.trim();
    if (!name) {
      infoBox("Enter pose name.");
      return;
    }
    const res = await window.operatorApi(
      `/processes/${encodeURIComponent(state.currentAssetId)}/poses`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, mode: "auto" }),
      }
    );
    if (!res.ok) {
      infoBox("Pose recording failed.");
      return;
    }
    byId("poseNameInput").value = "";
    await refreshPoses();
    infoBox("Pose recorded.");
  }

  async function deletePose() {
    if (!state.currentAssetId) return;
    const pose = byId("poseSelect").value;
    if (!pose) return;
    const res = await window.operatorApi(
      `/processes/${encodeURIComponent(state.currentAssetId)}/poses/${encodeURIComponent(
        pose
      )}`,
      { method: "DELETE" }
    );
    if (!res.ok) {
      infoBox("Pose delete failed.");
      return;
    }
    await refreshPoses();
    infoBox("Pose deleted.");
  }

  function readFileAsDataURL(file) {
    return new Promise((resolve, reject) => {
      const fr = new FileReader();
      fr.onload = () => resolve(String(fr.result || ""));
      fr.onerror = () => reject(new Error("file_read_failed"));
      fr.readAsDataURL(file);
    });
  }

  function extFromName(name) {
    const raw = String(name || "").toLowerCase();
    if (raw.endsWith(".jpg") || raw.endsWith(".jpeg")) return "jpg";
    if (raw.endsWith(".bmp")) return "png";
    return "png";
  }

  async function uploadTemplate() {
    if (!state.currentAssetId) {
      infoBox("Select an asset first.");
      return;
    }
    const objectId = byId("objectIdInput").value.trim();
    const templateName = byId("templateNameInput").value.trim();
    const fileInput = byId("templateFile");
    const file =
      fileInput && fileInput.files && fileInput.files.length ? fileInput.files[0] : null;
    if (!objectId || !templateName || !file) {
      infoBox("Object id, template label, and image are required.");
      return;
    }

    const image_b64 = await readFileAsDataURL(file);
    const payload = {
      object_id: objectId,
      template_name: templateName,
      image_b64,
      ext: extFromName(file.name),
    };
    const res = await window.operatorApi(
      `/processes/${encodeURIComponent(
        state.currentAssetId
      )}/objects/templates/upload`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }
    );

    if (!res.ok) {
      byId("assetInfo").textContent = JSON.stringify(
        res.body || { error: "upload_failed" },
        null,
        2
      );
      infoBox("Template upload failed.");
      return;
    }

    byId("assetInfo").textContent = JSON.stringify(res.body || {}, null, 2);
    await refreshObjects();
    infoBox("Template uploaded.");
  }
  async function startRun() {
    if (!state.currentTaskId) {
      infoBox("Select a task first.");
      return;
    }
    const res = await window.operatorApi(
      `/tasks/${encodeURIComponent(state.currentTaskId)}/runs/start`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ params: {} }),
      }
    );
    if (!res.ok || !res.body) {
      infoBox("Run start failed.");
      return;
    }
    state.currentRunId = res.body.run_id || "";
    byId("runInfo").textContent = JSON.stringify(res.body, null, 2);
    infoBox("Run started.");
    startRunPolling();
  }

  async function stopRun() {
    const runId = state.currentRunId;
    if (!runId) {
      infoBox("No active run id.");
      return;
    }
    const res = await window.operatorApi(
      `/runs/${encodeURIComponent(runId)}/stop`,
      { method: "POST" }
    );
    byId("runInfo").textContent = JSON.stringify(res.body || {}, null, 2);
    if (!res.ok) {
      infoBox("Run stop failed.");
      return;
    }
    infoBox("Run stop requested.");
  }

  async function refreshRunState() {
    if (!state.currentRunId) return;
    const res = await window.operatorApi(
      `/runs/${encodeURIComponent(state.currentRunId)}`,
      {},
      { silent: true }
    );
    if (!res.ok || !res.body) return;
    byId("runInfo").textContent = JSON.stringify(res.body, null, 2);
    const stateLabel = String(res.body.state || "").toLowerCase();
    const phaseLabel = String(res.body.phase || "").toLowerCase();
    setText(
      "runStatusText",
      phaseLabel && stateLabel === "running" ? phaseLabel : stateLabel || "idle"
    );
    setLed("runStatusLed", stateLabel === "running" ? "on" : "off");

    const events = await window.operatorApi(
      `/runs/${encodeURIComponent(state.currentRunId)}/timeline?limit=8`,
      {},
      { silent: true }
    );
    if (events.ok && events.body) {
      const list = events.body.events || [];
      byId("runTimeline").textContent = list
        .map((e) => `${e.event || "EVENT"} ${e.stage ? `(${e.stage})` : ""}`.trim())
        .join("\n");
    }
  }

  function startRunPolling() {
    if (state.runTimer) return;
    state.runTimer = setInterval(() => {
      refreshRunState();
    }, 1500);
  }

  function stopRunPolling() {
    if (!state.runTimer) return;
    clearInterval(state.runTimer);
    state.runTimer = null;
  }

  async function refreshSystemStatus() {
    const cam = await window.operatorApi("/camera/cameras", {}, { silent: true });
    if (cam.ok && cam.body) {
      const cams = Array.isArray(cam.body.cameras) ? cam.body.cameras : [];
      state.cameraIds = cams;
      setLed("cameraLed", cams.length ? "on" : "off");
      setText("cameraText", cams.length ? "on" : "off");
    } else {
      state.cameraIds = [];
      setLed("cameraLed", "error");
      setText("cameraText", "off");
    }

    const robot = await window.operatorApi("/robot/state", {}, { silent: true });
    if (!robot.ok || !robot.body) {
      setLed("robotLed", "error");
      setText("robotText", "off");
    } else {
      const mode = String(robot.body.mode || "").toUpperCase();
      const lastError = String(robot.body.last_error || "").toUpperCase();
      const unhealthy =
        !robot.body.connected ||
        mode.includes("DISCONNECT") ||
        mode.includes("ERROR") ||
        mode.includes("FAULT") ||
        lastError.includes("ERROR") ||
        lastError.includes("DISCONNECT") ||
        lastError.includes("COLLISION");
      setLed("robotLed", unhealthy ? "error" : "on");
      setText("robotText", unhealthy ? "off" : "on");
    }

    const vision = await window.operatorApi("/vision/cameras", {}, { silent: true });
    if (vision.ok && vision.body) {
      const running = !!vision.body.engine_running;
      const transport = String(vision.body.transport || "").toLowerCase();
      setLed("visionLed", running ? "on" : "off");
      setText("visionText", running ? (transport === "websocket" ? "blackwell" : "running") : "off");
    } else {
      setLed("visionLed", "error");
      setText("visionText", "off");
    }

    await refreshVisionTransport();

    const health = await window.operatorApi("/health", {}, { silent: true });
    const healthStatus = health && health.body ? health.body.status : "";
    if (health.ok && String(healthStatus || "").toLowerCase() === "alive") {
      setLed("serverLed", "on");
      setText("serverText", "on");
    } else {
      setLed("serverLed", "error");
      setText("serverText", "off");
    }

    renderFlowPreview();
  }

  async function refreshVisionTransport() {
    const select = byId("visionTransportSelect");
    if (!select) return;
    const res = await window.operatorApi("/runtime/vision-transport", {}, { silent: true });
    if (!res.ok || !res.body) return;
    const transport = String(res.body.transport || "zmq").toLowerCase();
    select.value = transport === "websocket" ? "websocket" : "zmq";
  }

  function bindEvents() {
    byId("refreshWorkspaceBtn").addEventListener("click", async () => {
      await reloadWorkspace();
      infoBox("Workspace refreshed.");
    });

    byId("stationSelect").addEventListener("change", async (e) => {
      state.currentStationId = e.target.value;
      await refreshAssets();
      await refreshTasks();
      await refreshPoses();
      await refreshObjects();
      await loadTaskFlow();
    });

    byId("assetSelect").addEventListener("change", async (e) => {
      state.currentAssetId = e.target.value;
      const active = selectedAsset();
      byId("assetName").value = (active && active.name) || "";
      await refreshTasks();
      await refreshPoses();
      await refreshObjects();
      await loadTaskFlow();
    });
    byId("taskSelect").addEventListener("change", async (e) => {
      state.currentTaskId = e.target.value;
      const task = selectedTask();
      if (task) setTaskType(task.task_type || state.currentTaskType);
      await loadTaskFlow();
    });

    byId("taskTypeSelect").addEventListener("change", (e) => {
      setTaskType(e.target.value, { userSelected: true });
    });

    byId("saveAssetBtn").addEventListener("click", saveAssetName);
    byId("createTaskBtn").addEventListener("click", createTask);

    byId("recordPoseBtn").addEventListener("click", recordPose);
    byId("deletePoseBtn").addEventListener("click", deletePose);
    byId("refreshPosesBtn").addEventListener("click", refreshPoses);

    byId("uploadTemplateBtn").addEventListener("click", uploadTemplate);

    const dummyAddPoseBtn = byId("dummyAddPoseBtn");
    if (dummyAddPoseBtn) dummyAddPoseBtn.addEventListener("click", addDummyPoseStep);

    const dummyApplyProgramBtn = byId("dummyApplyProgramBtn");
    if (dummyApplyProgramBtn) dummyApplyProgramBtn.addEventListener("click", applyDummyProgramToWorkspace);

    const dummyClearProgramBtn = byId("dummyClearProgramBtn");
    if (dummyClearProgramBtn) {
      dummyClearProgramBtn.addEventListener("click", () => {
        setDummyProgram([]);
        infoBox("Dummy Testing sequence cleared.");
      });
    }

    const dummyPoseSequence = byId("dummyPoseSequence");
    if (dummyPoseSequence) {
      dummyPoseSequence.addEventListener("click", (event) => {
        const target = event.target;
        if (!target || !target.dataset) return;
        const index = toSafeInteger(target.dataset.index, -1, -1);
        const action = String(target.dataset.action || "");
        if (action === "up") moveDummyProgramStep(index, -1);
        if (action === "down") moveDummyProgramStep(index, 1);
        if (action === "remove") removeDummyProgramStep(index);
      });
    }

    byId("saveFlowBtn").addEventListener("click", saveFlow);
    byId("resetFlowBtn").addEventListener("click", () => {
      buildWorkspaceFromFlow([]);
      infoBox("Flow cleared.");
    });

    byId("fitFlowBtn").addEventListener("click", () => {
      if (!state.workspace) return;
      state.workspace.zoomToFit();
    });

    byId("normalizeFlowBtn").addEventListener("click", () => {
      normalizeFlowLayout();
      renderFlowPreview();
    });

    byId("startRunBtn").addEventListener("click", startRun);
    byId("stopRunBtn").addEventListener("click", stopRun);

    const visionTransportSelect = byId("visionTransportSelect");
    if (visionTransportSelect && !state.boundVisionTransport) {
      visionTransportSelect.addEventListener("change", async (event) => {
        const transport = String(event.target.value || "zmq");
        const res = await window.operatorApi("/runtime/vision-transport", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ transport }),
        });
        if (!res.ok) {
          await refreshVisionTransport();
          infoBox("Vision transport update failed.");
          return;
        }
        infoBox(transport === "websocket" ? "Vision mode: blackwell." : "Vision mode: local.");
        await refreshSystemStatus();
      });
      state.boundVisionTransport = true;
    }
  }

  async function reloadWorkspace() {
    await refreshStations();
    await refreshTaskTypes();
    await refreshAssets();
    await refreshTasks();
    await refreshPoses();
    await refreshObjects();
    await loadTaskFlow();
  }

  async function init() {
    bindEvents();
    await refreshTaskTypes();
    initBlocklyWorkspace();
    await reloadWorkspace();
    await refreshSystemStatus();
    state.statusTimer = setInterval(refreshSystemStatus, 3000);
    startRunPolling();
    infoBox("Operator UI ready.");
  }

  window.addEventListener("beforeunload", () => {
    if (state.statusTimer) clearInterval(state.statusTimer);
    stopRunPolling();
    if (state.resizeHandler) {
      window.removeEventListener("resize", state.resizeHandler);
      state.resizeHandler = null;
    }
    if (state.workspace) {
      state.workspace.dispose();
      state.workspace = null;
    }
  });

  document.addEventListener("DOMContentLoaded", init);
})();
