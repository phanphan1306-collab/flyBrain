function analyze_mb_learning(recordPath, rawPath, outputDir)
%ANALYZE_MB_LEARNING Offline analysis of one MB_ant training record.
%   Reads Record and Raw_images only. Writes figures and CSVs to outputDir.
%   Requires no GPU and does not modify the source MAT files.

if nargin ~= 3
    error('Usage: analyze_mb_learning(recordPath, rawPath, outputDir)');
end
if ~exist(recordPath, 'file') || ~exist(rawPath, 'file')
    error('Input MAT file missing: %s or %s', recordPath, rawPath);
end
recordData = load(recordPath, 'Record');
rawData = load(rawPath, 'Raw_images');
if ~isfield(recordData, 'Record') || ~isstruct(recordData.Record)
    error('Record struct missing from %s', recordPath);
end
if ~isfield(rawData, 'Raw_images') || ~isstruct(rawData.Raw_images)
    error('Raw_images struct missing from %s', rawPath);
end
Record = recordData.Record;
Raw_images = rawData.Raw_images;
required = {'PN_activity', 'KC_activity', 'EN_activity'};
for k = 1:numel(required)
    if ~isfield(Record, required{k})
        error('Record.%s is missing', required{k});
    end
end
if ~isfield(Raw_images, 'heading') || ~isfield(Raw_images, 'img_pos')
    error('Raw_images.heading or Raw_images.img_pos is missing');
end
N = numel(Raw_images.heading);
if N < 1 || size(Raw_images.img_pos, 1) ~= N + 1 || size(Raw_images.img_pos, 2) ~= 2
    error('Raw_images heading/img_pos dimensions do not match');
end
PN = double(Record.PN_activity(:));
KC = double(Record.KC_activity(:));
EN = double(Record.EN_activity(:));
if numel(PN) ~= 3*N || numel(KC) ~= 3*N || numel(EN) ~= 3*N
    error('PN/KC/EN activity lengths must each equal 3*numel(heading)');
end
if any(~isfinite([PN; KC; EN])) || any([PN; KC; EN] < 0)
    error('PN/KC/EN activities contain NaN, Inf or negative values');
end
numKC = 20000; % ant_route_following_test.m:82; not inferred from KC_fired_total
if any(KC > numKC)
    error('KC_activity exceeds the configured numKC=20000');
end
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

pre = EN(1:N);
training = EN(N+1:2*N);
post = EN(2*N+1:3*N);
delta = post - pre;
zeroBoth = (pre == 0) & (post == 0);
kcSparse = reshape(KC / numKC, N, 3);

fig = figure('Visible', 'off', 'Color', 'w');
plot(1:N, pre, 'b.-', 1:N, post, 'r.-');
xlabel('Image index on trained route');
ylabel('EN spikes / trial');
title('EN response to the same images before and after learning');
legend('Pre-training', 'Post-training', 'Location', 'best');
grid on;
print(fig, fullfile(outputDir, 'en_before_after.png'), '-dpng', '-r150');
close(fig);

fig = figure('Visible', 'off', 'Color', 'w');
plot(1:N, kcSparse(:,1), 'b-', 1:N, kcSparse(:,2), 'k-', ...
    1:N, kcSparse(:,3), 'r-');
xlabel('Image index on trained route');
ylabel('Fraction of KC neurons active / trial');
title('KC sparsity: KC activity divided by 20000');
legend('Pre-training', 'Training', 'Post-training', 'Location', 'best');
grid on;
print(fig, fullfile(outputDir, 'kc_sparsity.png'), '-dpng', '-r150');
close(fig);

f = fopen(fullfile(outputDir, 'learning_per_image.csv'), 'w');
if f < 0, error('Cannot write learning_per_image.csv'); end
fprintf(f, 'image_index,en_pre_spikes,en_training_spikes,en_post_spikes,en_delta_post_minus_pre,kc_pre_fraction,kc_training_fraction,kc_post_fraction,en_zero_both\n');
for i = 1:N
    fprintf(f, '%d,%.15g,%.15g,%.15g,%.15g,%.15g,%.15g,%.15g,%d\n', ...
        i, pre(i), training(i), post(i), delta(i), ...
        kcSparse(i,1), kcSparse(i,2), kcSparse(i,3), zeroBoth(i));
end
fclose(f);

f = fopen(fullfile(outputDir, 'learning_summary.csv'), 'w');
if f < 0, error('Cannot write learning_summary.csv'); end
fprintf(f, 'measure,phase,n,mean,median,std_sample\n');
labels = {'pre', 'training', 'post'};
activities = {PN, KC, EN};
names = {'PN_active_neurons', 'KC_active_neurons', 'EN_spikes'};
for a = 1:3
    values = activities{a};
    for phase = 1:3
        v = values((phase-1)*N+1:phase*N);
        fprintf(f, '%s,%s,%d,%.15g,%.15g,%.15g\n', ...
            names{a}, labels{phase}, N, mean(v), median(v), std(v));
    end
end
for phase = 1:3
    v = kcSparse(:,phase);
    fprintf(f, 'KC_active_fraction,%s,%d,%.15g,%.15g,%.15g\n', ...
        labels{phase}, N, mean(v), median(v), std(v));
end
fprintf(f, 'EN_delta_post_minus_pre,paired,%d,%.15g,%.15g,%.15g\n', ...
    N, mean(delta), median(delta), std(delta));
fprintf(f, 'EN_zero_both_fraction,paired,%d,%.15g,%.15g,%.15g\n', ...
    N, mean(zeroBoth), median(double(zeroBoth)), std(double(zeroBoth)));
fclose(f);

f = fopen(fullfile(outputDir, 'learning_validation.txt'), 'w');
if f < 0, error('Cannot write learning_validation.txt'); end
fprintf(f, 'record=%s\nraw_images=%s\nimages=%d\n', recordPath, rawPath, N);
fprintf(f, 'activity_length_each=%d (expected %d)\n', numel(EN), 3*N);
fprintf(f, 'EN_zero_both_count=%d\n', sum(zeroBoth));
fprintf(f, 'numKC=20000 from ant_route_following_test.m:82\n');
fprintf(f, 'KC_activity counts neurons with >=1 spike, not total KC spikes.\n');
fprintf(f, 'KC_fired_total excluded: original code has inconsistent indexing.\n');
fprintf(f, 'Seed and pre-training weight snapshot are not stored in Record.\n');
fprintf(f, 'Lower EN spike count is a model familiarity proxy, not proof of learning by itself.\n');
fclose(f);
end
