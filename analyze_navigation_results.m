function analyze_navigation_results(navPath, rawPath, outputDir)
%ANALYZE_NAVIGATION_RESULTS Offline analysis of one MB_ant navigation run.
%   Reads navigation_result and Raw_images only; no GPU or simulation call.

if nargin ~= 3
    error('Usage: analyze_navigation_results(navPath, rawPath, outputDir)');
end
if ~exist(navPath, 'file') || ~exist(rawPath, 'file')
    error('Input MAT file missing: %s or %s', navPath, rawPath);
end
navData = load(navPath, 'navigation_result');
rawData = load(rawPath, 'Raw_images');
if ~isfield(navData, 'navigation_result') || ~isstruct(navData.navigation_result)
    error('navigation_result struct missing from %s', navPath);
end
if ~isfield(rawData, 'Raw_images') || ~isstruct(rawData.Raw_images)
    error('Raw_images struct missing from %s', rawPath);
end
nav = navData.navigation_result;
raw = rawData.Raw_images;
required = {'current_position', 'step_record', 'perf_measure', ...
    'error_location', 'error_rate', 'dis_threshold'};
for k = 1:numel(required)
    if ~isfield(nav, required{k})
        error('navigation_result.%s is missing', required{k});
    end
end
if ~isfield(raw, 'img_pos') || ~isfield(raw, 'heading')
    error('Raw_images.img_pos or Raw_images.heading is missing');
end
route = double(raw.img_pos);
position = double(nav.current_position);
steps = double(nav.step_record);
perf = double(nav.perf_measure(:));
threshold = double(nav.dis_threshold);
recordedErrorRate = double(nav.error_rate);
if size(route, 2) ~= 2 || size(route, 1) ~= numel(raw.heading)+1 ...
        || size(position, 2) ~= 2 || size(position, 1) < 2
    error('Route or current_position has invalid dimensions');
end
if size(steps, 1) < 3 || size(steps, 2) ~= size(position, 1)
    error('step_record must have at least 3 rows and one column per saved position');
end
if numel(perf) < size(position, 1)-1
    error('perf_measure is shorter than the saved navigation steps');
end
if ~isempty(nav.error_location) && size(nav.error_location, 2) ~= 2
    error('error_location must be empty or N x 2');
end
numericFields = [route(:); position(:); steps(:); perf(:); threshold; recordedErrorRate];
if any(~isfinite(numericFields)) || threshold <= 0
    error('Navigation data contains NaN, Inf or invalid threshold');
end
if any(perf ~= 0 & perf ~= 1) || any(steps(3,:) ~= 0 & steps(3,:) ~= 1)
    error('perf_measure and step_record(3,:) must be binary');
end
if ~isempty(nav.error_location) && any(~isfinite(nav.error_location(:)))
    error('error_location contains NaN or Inf');
end
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

numTransitions = size(position, 1)-1; % step_count in the original crop
scanFlag = steps(3, 1:numTransitions) == 1;
segmentLength = sqrt(sum(diff(position, 1, 1).^2, 2));
routeLength = sum(sqrt(sum(diff(route, 1, 1).^2, 2)));
commandedLength = sum(segmentLength(scanFlag(:)));
resetJumpLength = sum(segmentLength(~scanFlag(:)));
numScans = sum(scanFlag);
numAssists = sum(perf);
numErrorLocations = size(nav.error_location, 1);
computedErrorRate = numAssists / numTransitions;
distanceToNest = norm(position(end,:) - route(end,:));
withinNestRadius = distanceToNest <= threshold;
commandedEndpoints = position([false; scanFlag(:)], :);
offRoute = distanceToPolyline(commandedEndpoints, route);
if isempty(offRoute)
    meanOffRoute = NaN;
    maxOffRoute = NaN;
else
    meanOffRoute = mean(offRoute);
    maxOffRoute = max(offRoute);
end

fig = figure('Visible', 'off', 'Color', 'w');
hold on;
hRoute = plot(route(:,1), route(:,2), 'r-', 'LineWidth', 1.5);
hMove = plot(NaN, NaN, 'b-', 'LineWidth', 1.1);
hReset = plot(NaN, NaN, 'k--', 'LineWidth', 0.8);
for i = 1:numTransitions
    if scanFlag(i)
        plot(position(i:i+1,1), position(i:i+1,2), 'b-', 'LineWidth', 1.1);
    else
        plot(position(i:i+1,1), position(i:i+1,2), 'k--', 'LineWidth', 0.8);
    end
end
hStart = plot(route(1,1), route(1,2), 'go', 'MarkerFaceColor', 'g');
hNest = plot(route(end,1), route(end,2), 'ro', 'MarkerFaceColor', 'r');
if numErrorLocations > 0
    hAssist = plot(nav.error_location(:,1), nav.error_location(:,2), 'g*');
    legend([hRoute hMove hReset hStart hNest hAssist], ...
        {'Trained route', 'Commanded movement', 'Assisted reset jump', ...
        'Feeder', 'Nest', 'Reset destination'}, 'Location', 'best');
else
    legend([hRoute hMove hReset hStart hNest], ...
        {'Trained route', 'Commanded movement', 'Assisted reset jump', ...
        'Feeder', 'Nest'}, 'Location', 'best');
end
axis equal;
xlabel('X (m)'); ylabel('Y (m)');
title('MB ant route: movement and assisted resets');
grid on;
print(fig, fullfile(outputDir, 'route_overlay.png'), '-dpng', '-r150');
close(fig);

f = fopen(fullfile(outputDir, 'navigation_summary.csv'), 'w');
if f < 0, error('Cannot write navigation_summary.csv'); end
fprintf(f, 'metric,value,unit\n');
fprintf(f, 'saved_transitions,%d,count\n', numTransitions);
fprintf(f, 'scanned_movement_steps,%d,count\n', numScans);
fprintf(f, 'assisted_resets,%d,count\n', numAssists);
fprintf(f, 'error_locations_saved,%d,count\n', numErrorLocations);
fprintf(f, 'error_rate_recorded,%.15g,fraction\n', recordedErrorRate);
fprintf(f, 'error_rate_recomputed,%.15g,fraction\n', computedErrorRate);
fprintf(f, 'mean_off_route_commanded_endpoint,%.15g,m\n', meanOffRoute);
fprintf(f, 'max_off_route_commanded_endpoint,%.15g,m\n', maxOffRoute);
fprintf(f, 'reference_route_length,%.15g,m\n', routeLength);
fprintf(f, 'commanded_movement_length,%.15g,m\n', commandedLength);
fprintf(f, 'assisted_reset_jump_length,%.15g,m\n', resetJumpLength);
fprintf(f, 'distance_last_saved_position_to_nest,%.15g,m\n', distanceToNest);
fprintf(f, 'last_position_within_nest_radius,%d,boolean\n', withinNestRadius);
fprintf(f, 'nest_reached_without_assistance,undetermined,status\n');
fclose(f);

f = fopen(fullfile(outputDir, 'navigation_validation.txt'), 'w');
if f < 0, error('Cannot write navigation_validation.txt'); end
fprintf(f, 'navigation=%s\nraw_images=%s\n', navPath, rawPath);
fprintf(f, 'positions=%d\nreference_positions=%d\n', size(position,1), size(route,1));
fprintf(f, 'recorded_error_rate=%.15g\nrecomputed_error_rate=%.15g\n', ...
    recordedErrorRate, computedErrorRate);
if abs(recordedErrorRate-computedErrorRate) > 1e-10
    fprintf(f, 'WARNING: recorded error_rate differs from sum(perf_measure)/saved_transitions.\n');
end
if numAssists ~= numErrorLocations
    fprintf(f, 'WARNING: perf_measure count differs from error_location row count.\n');
end
fprintf(f, 'Off-route distance uses commanded movement endpoints and nearest reference polyline segments.\n');
fprintf(f, 'Commanded length excludes reset jumps; total plotted path includes them.\n');
fprintf(f, 'error_location stores reset destinations, not departure positions.\n');
fprintf(f, 'Original MAT has no stop reason or saved random seed.\n');
fprintf(f, 'Near-nest last position is not proof of unassisted success.\n');
fclose(f);
end

function distances = distanceToPolyline(points, route)
distances = zeros(size(points,1), 1);
for i = 1:size(points,1)
    best = Inf;
    for j = 1:size(route,1)-1
        a = route(j,:);
        b = route(j+1,:);
        ab = b-a;
        lengthSquared = sum(ab.^2);
        if lengthSquared == 0
            projection = a;
        else
            t = max(0, min(1, sum((points(i,:)-a).*ab)/lengthSquared));
            projection = a+t*ab;
        end
        best = min(best, norm(points(i,:)-projection));
    end
    distances(i) = best;
end
end
