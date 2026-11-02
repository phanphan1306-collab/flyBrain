function logPath = run_snn_heading_scan(antIds, outputDir, opts)
%RUN_SNN_HEADING_SCAN Train the original PN/KC/EN SNN, then scan held-out poses.
%   logPath = run_snn_heading_scan([1 2], outputDir)
%   Ant routes are split by identity: first half validation, rest test.
%   This requires MATLAB with GPU/Parallel and Image Processing toolboxes.
%   Run a small pilot first; the original 20,000-KC network is expensive.
%
%   The simulator explicitly places a virtual agent at midpoints between
%   trained route images and assigns its body heading from the route tangent.
%   A biased/noisy virtual gyro is derived from successive assigned headings.
%   Thus heading_truth is simulator state, not a measured ant body heading.
%   Visual scores are actual EN spike counts from the original SNN code.

if nargin < 1 || isempty(antIds), antIds = [1 2]; end
if nargin < 2 || isempty(outputDir)
    outputDir = fullfile(fileparts(mfilename('fullpath')), '..', 'phase2', 'results', 'snn');
end
if nargin < 3 || isempty(opts), opts = struct(); end
if numel(antIds) < 2 || numel(unique(antIds)) ~= numel(antIds) || ...
        any(antIds < 1) || any(antIds > 15) || any(mod(antIds, 1) ~= 0)
    error('Use at least two distinct ant IDs from 1..15 for route-held-out validation/test.');
end
opts = setDefault(opts, 'seed', 20261027);
opts = setDefault(opts, 'route_stride', 3);
opts = setDefault(opts, 'max_frames', 25);
opts = setDefault(opts, 'scan_half_range_deg', 60);
opts = setDefault(opts, 'scan_step_deg', 10);
opts = setDefault(opts, 'lateral_offset_m', 0.03);
opts = setDefault(opts, 'gyro_bias_deg_per_frame', 0.8);
opts = setDefault(opts, 'gyro_noise_std_deg', 0.2);
opts = setDefault(opts, 'initial_error_deg', 5);
opts = setDefault(opts, 'context_tolerance_indices', 4);
if opts.route_stride < 1 || mod(opts.route_stride, 1) ~= 0 || ...
        opts.max_frames < 2 || mod(opts.max_frames, 1) ~= 0 || ...
        opts.gyro_noise_std_deg < 0 || opts.context_tolerance_indices < 0 || ...
        opts.scan_step_deg <= 0 || ...
        opts.scan_half_range_deg <= 0 || ...
        mod(2 * opts.scan_half_range_deg, opts.scan_step_deg) ~= 0
    error('Invalid scan/route options.');
end

projectDir = fileparts(fileparts(mfilename('fullpath')));
sourceDir = fullfile(projectDir, 'source_data');
addpath(sourceDir);
world = load(fullfile(sourceDir, 'antview', 'world5000_gray.mat'), ...
             'X', 'Y', 'Z', 'colp');
X = world.X; Y = world.Y; Z = world.Z; colp = world.colp;
if ~exist(outputDir, 'dir'), mkdir(outputDir); end
logPath = fullfile(outputDir, sprintf('snn_heading_seed%d.jsonl', opts.seed));
partialPath = [logPath, '.partial'];
if exist(logPath, 'file') || exist(partialPath, 'file')
    error('Output or partial run already exists: %s. Choose a new directory or seed.', logPath);
end
fid = fopen(partialPath, 'w');
if fid < 0, error('Cannot create %s', partialPath); end
cleanupFile = onCleanup(@() fclose(fid));

numPN = 360; numKC = 20000; numEN = 1;
C_I_PN_var = 5250; g_PN_KC = 0.25; g_KC_EN = 2.0;
interval = 50; dt = 1;
eye_height = 0.01; resolution = 4; hfov = 296;
nValidation = floor(numel(antIds) / 2);

for antIndex = 1:numel(antIds)
    ant = antIds(antIndex);
    rng(opts.seed + ant, 'twister');
    file = fullfile(sourceDir, 'result', ...
                    sprintf('Raw_images_ant%d_route1.mat', ant));
    loaded = load(file, 'Raw_images');
    raw = loaded.Raw_images;
    numTrain = numel(raw.heading);
    route = double(raw.img_pos);
    if size(route, 1) ~= numTrain + 1 || size(route, 2) ~= 2
        error('Ant %d has invalid route dimensions.', ant);
    end

    % New controlled training run. Archived Record*.mat cannot reproduce
    % connectivity because the original random seed was not saved.
    connection_generator;
    null_input = gpuArray.zeros(numPN, 1);
    connection_PN_KC = gpuArray(connection_PN_KC);
    connection_KC_EN = gpuArray(connection_KC_EN);
    weight_matrix_KC_EN = connection_KC_EN * g_KC_EN;
    kcTemplates = false(numKC, numTrain);
    templateCounts = zeros(1, numTrain);
    reward = 1;
    for trainIndex = 1:numTrain
        input = gpuArray(panoramaInput(raw.position(trainIndex).origin, ...
                                      numPN, C_I_PN_var));
        GPU_network_fun;
        code = logical(gather(any(spike_time_KC, 2)));
        kcTemplates(:, trainIndex) = code;
        templateCounts(trainIndex) = nnz(code);
    end
    reward = 0;

    selected = 1:opts.route_stride:(numTrain - 1);
    selected = selected(1:min(numel(selected), opts.max_frames));
    if numel(selected) < 2, error('Ant %d has too few held-out poses.', ant); end
    if antIndex <= nValidation, split = 'validation'; else, split = 'test'; end
    runId = sprintf('snn_ant%d_seed%d', ant, opts.seed);
    gyroHeading = wrapDeg(double(raw.heading(selected(1))) + opts.initial_error_deg);
    previousTruth = double(raw.heading(selected(1)));
    framePose = zeros(numel(selected), 2);
    frameTruth = zeros(numel(selected), 1);
    frameGyro = zeros(numel(selected), 1);

    for frameIndex = 1:numel(selected)
        routeIndex = selected(frameIndex);
        truth = double(raw.heading(routeIndex));
        pose = (route(routeIndex, :) + route(routeIndex + 1, :)) / 2;
        pose = pose + opts.lateral_offset_m * [-sind(truth), cosd(truth)];
        if frameIndex == 1
            gyroDelta = 0;
        else
            trueDelta = wrapDeg(truth - previousTruth);
            gyroDelta = trueDelta + opts.gyro_bias_deg_per_frame + ...
                        opts.gyro_noise_std_deg * randn();
            gyroHeading = wrapDeg(gyroHeading + gyroDelta);
        end
        previousTruth = truth;
        framePose(frameIndex, :) = pose;
        frameTruth(frameIndex) = truth;
        frameGyro(frameIndex) = gyroDelta;

        sectorAngles = gyroHeading + ...
            (-opts.scan_half_range_deg:opts.scan_step_deg:opts.scan_half_range_deg);
        sectorAngles = arrayfun(@wrapDeg, sectorAngles);
        enScores = zeros(1, numel(sectorAngles));
        placeMatches = zeros(1, numel(sectorAngles));
        for sectorIndex = 1:numel(sectorAngles)
            picture = ImgGrabber(pose(1), pose(2), eye_height, ...
                sectorAngles(sectorIndex), X, Y, Z, colp, hfov, resolution);
            input = gpuArray(panoramaInput(picture, numPN, C_I_PN_var));
            GPU_network_fun;
            enScores(sectorIndex) = double(gather(sum(spike_time_EN(:))));
            code = logical(gather(any(spike_time_KC, 2)));
            active = find(code);
            overlap = sum(kcTemplates(active, :), 1);
            union = templateCounts + numel(active) - overlap;
            [~, matchIndex] = max(overlap ./ max(union, 1));
            placeMatches(sectorIndex) = matchIndex;
        end

        [~, bestSector] = min(enScores); % MATLAB's first minimum rule
        matchedPlace = placeMatches(bestSector);
        odomPlace = routeIndex + 0.5 * randn(); % independent virtual odometry
        contextOK = abs(matchedPlace - odomPlace) <= opts.context_tolerance_indices;
        observation = struct();
        observation.schema_version = 1;
        observation.run_id = runId;
        observation.route_id = sprintf('ant%d_route1', ant);
        observation.split = split;
        observation.origin = 'simulator';
        observation.condition = 'snn_heldout_midpoint';
        observation.frame = frameIndex - 1;
        observation.timestamp_s = (frameIndex - 1) * 0.1;
        observation.gyro_delta_deg = gyroDelta;
        observation.heading_truth_deg = truth;
        observation.sector_angles_deg = sectorAngles;
        observation.familiarity_scores = enScores;
        observation.visual_source = 'snn_en_scan';
        observation.context_ok = logical(contextOK);
        observation.context_source = 'kc_place_match_vs_virtual_odometry';
        observation.odom_place_id = odomPlace;
        observation.matched_place_id = matchedPlace;
        observation.event_type = 'heldout_midpoint';
        fprintf(fid, '%s\n', jsonencode(observation));
        if mod(frameIndex, 5) == 0 || frameIndex == numel(selected)
            fprintf('Ant %d: SNN scan %d/%d\n', ant, frameIndex, numel(selected));
        end
    end
    metadata = struct('ant', ant, 'split', split, 'seed', opts.seed + ant, ...
        'options', opts, 'trained_images', numTrain, 'heldout_route_indices', selected, ...
        'frame_pose_m', framePose, 'heading_truth_deg', frameTruth, ...
        'gyro_delta_deg', frameGyro, 'kc_templates_active_count', templateCounts);
    model = struct('numPN', numPN, 'numKC', numKC, 'numEN', numEN, ...
        'connection_PN_KC', sparse(gather(connection_PN_KC)), ...
        'weight_matrix_KC_EN', gather(weight_matrix_KC_EN), ...
        'kc_training_templates', kcTemplates);
    save(fullfile(outputDir, sprintf('%s_metadata.mat', runId)), ...
         'metadata', 'model', '-v7.3');
    fprintf('Ant %d: %d SNN scan frames (%s)\n', ant, numel(selected), split);
end
clear cleanupFile;
[moved, message] = movefile(partialPath, logPath);
if ~moved, error('Could not finalize JSONL log: %s', message); end
fprintf('Wrote SNN heading log: %s\n', logPath);
end

function values = panoramaInput(image, numPN, scale)
resized = imresize(image, [10, 36]);
inverted = 1 - double(resized) / 255;
equalized = adapthisteq(inverted);
values = reshape(equalized, numPN, 1);
normValue = sqrt(sum(values .^ 2));
if normValue <= 0, error('Zero-norm panorama input.'); end
values = values / normValue * scale;
end

function value = wrapDeg(angle)
value = mod(angle + 180, 360) - 180;
end

function options = setDefault(options, name, value)
if ~isfield(options, name), options.(name) = value; end
end
