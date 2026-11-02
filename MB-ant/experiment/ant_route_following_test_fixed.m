% This is the file to do all the ant route following tests, based on 
% different ant routes data.
% it will automatically load ant data and save corresponding results.
% it has a Format that includes three parts:
% --> 1. generate route image (involve load specific antdata, generate 
%        correct headings and on-route images);
% --> 2. train a naive network with those images;
% --> 3. test the trained network with the simulated world by releasing 
%        the 'agent' ant from the feeding site, and allow the agent to 
%        scan a wide angular space for a heading decision, and move 
%        forward according to this heading direciton with a fixed step size


% Fei Peng, 21/11/2013, fei.peng@qmul.ac.uk

clear all; close all; clc; reset(gpuDevice);
% load two overall data:
load('./antview/AntData.mat');
load('./antview/world5000_gray.mat');
output_dir = '../experiment_results';
if ~exist(output_dir, 'dir'), mkdir(output_dir); end

% control the random seed
random_seed = randi(1000000);

count = 1;

% main loop: test for all the 15 different ant routes 

for i_ant = [Ant1, Ant2, Ant3, Ant4, Ant5, Ant6, Ant7, Ant8, Ant9, Ant10, ...
              Ant11, Ant12, Ant13, Ant14, Ant15] % 15 routes in total
    % apply the random seed
    rng(random_seed);

%-------------------***     1. image generation     ***-------------------%
   
    temp_route = i_ant.InwardRouteData.Route1.One_cm_control_points/100;
    % sample an image every 10 cm
    img_separation = 10; % [cm]
    img_limit = floor(size(temp_route,1)/10)*10; 
    img_pos = temp_route(1:img_separation:img_limit,:); % snapshot position
    numPos = size(img_pos,1)-1;
    heading = zeros(numPos,1); % initialise angles of all the snapshots
    
    % parameters for invoking ImgGrabber
    eye_height = 0.01; % [m]
    resolution = 4; % [degrees/pixel]
    hfov = 296; % [degrees]
    % scan_range = 10; % [degrees]
    %  scan_step = 2; % [degrees]
    Raw_images = struct;
    
    % Get Headings for num_pos positions
    for i = 1:numPos
        heading(i) = atan2(img_pos(i+1,2)-img_pos(i,2),img_pos(i+1,1)-img_pos(i,1))*180/pi;
    end
    % Round or ceil heading into recent even number:
    for i = 1:numPos
        if mod(floor(heading(i)), 2) == 0
            heading(i) = floor(heading(i));
        else
            heading(i) = ceil(heading(i));
        end
    end
    % according to adjusted heading, update all the img_pos, except the 
    % first position.
    step_size = 0.1; %[m]
    for i = 1:numPos
        img_pos(i+1,:) = img_pos(i,:) + [cosd(heading(i)), sind(heading(i))]*step_size;
    end
    for i = 1:numPos
        Raw_images.position(i).origin = ImgGrabber(img_pos(i,1), ...
            img_pos(i,2), eye_height, heading(i), X, Y, Z, colp, hfov, resolution);
    end
    Raw_images.img_pos = img_pos;
    Raw_images.heading = heading;
    save(fullfile(output_dir, sprintf('Raw_images_ant%d_route1.mat',count)),'Raw_images');
    close all;


%------------------***     2. Train the network     ***-------------------%
    
    % define the network parameters
    numPN = 360; numKC = 20000; numEN = 1; % archtecture
    C_I_PN_var =5250; % input scaling parameter
    g_PN_KC = 0.25; g_KC_EN = 2.0; % initialise max synaptic conductances
    interval = 50; dt = 1; numTrain = numPos;
    connection_generator; % generate connection matrix for PN-KC and KC-EN
    % convert the connection matrices into matlab-CUDA gpuArray
    null_input = gpuArray.zeros(numPN,1);
    connection_PN_KC = gpuArray(connection_PN_KC);
    connection_KC_EN = gpuArray(connection_KC_EN);

    % setup connection and weight matrix for KC-EN
    weight_matrix_KC_EN = connection_KC_EN*g_KC_EN;
    training_inputs = zeros(numPN, numTrain);
    
    for img = 1:numTrain, % for each image in the training set       
        % image pre-prosessing: reverse & adaptive histgram equalisation
        temp_img_1 = Raw_images.position(img).origin;
        temp_img_2 = imresize(temp_img_1, [10, 36]);
        temp_img_3 = 1-double(temp_img_2)/255;
        temp_img_4 = adapthisteq(temp_img_3);
        temp_img_5 = reshape(temp_img_4, numPN, 1);
        training_inputs(:,img) = temp_img_5;
    end
    
    % normalisation 
    training_inputs = training_inputs./(repmat(sqrt(sum(training_inputs.^2)),numPN,1));
    % scale input by parameter C_I_PN_var
    training_inputs = training_inputs*C_I_PN_var;
    training_inputs = gpuArray(training_inputs);
   
    clear Record; clear PN_activity KC_activity EN_activity

    % initialize recordings in gpuArray
    PN_activity = gpuArray.zeros(numTrain*3, 1);
    KC_activity = gpuArray.zeros(numTrain*3, 1);
    EN_activity = gpuArray.zeros(numTrain*3, 1);
    KC_fired_total = gpuArray.zeros(numKC, numTrain*3);
    
    % Phase1 --> Pre-Training phase
    for n_train_pre = 1:numTrain,
        n_train_pre
        % no reward so no learning
        reward = 0;
        % call the main network file
        input = training_inputs(:,n_train_pre); % input to PN
        GPU_network_fun; % call the network
        PN_ind = find(sum(spike_time_PN, 2)>0);
        PN_count = length(PN_ind); % how many PN fired
        KC_ind = find(sum(spike_time_KC, 2)>0);
        KC_count = length(KC_ind); % how many KC fired
        KC_fired_total(KC_ind, n_train_pre) = 1;
        EN_count = sum(sum(spike_time_EN)); % how many spikes EN generated
        PN_activity(n_train_pre) = PN_count;
        KC_activity(n_train_pre) = KC_count;
        EN_activity(n_train_pre) = EN_count;
    end
    
    % Phase2 --> Training phase
    for n_train = 1:numTrain,
        n_train
        % reward - learning is allowed
        reward = 1;
        % call the main network file
        input = training_inputs(:,n_train); % input to PN
        GPU_network_fun; % call the network
        PN_ind = find(sum(spike_time_PN, 2)>0);
        PN_count = length(PN_ind); % how many PN fired
        KC_ind = find(sum(spike_time_KC, 2)>0);
        KC_count = length(KC_ind); % how many KC fired
        KC_fired_total(KC_ind, numTrain + n_train) = 1;
        EN_count = sum(sum(spike_time_EN)); % how many spikes EN generated
        PN_activity(numTrain + n_train) = PN_count;
        KC_activity(numTrain + n_train) = KC_count;
        EN_activity(numTrain + n_train) = EN_count;
    end
    
    % Phase3 --> Pre-Training phase
    for n_train_post = 1:numTrain,
        n_train_post
        % no reward so no learning
        reward = 0;
        % call the main network file
        input = training_inputs(:,n_train_post); % input to PN
        GPU_network_fun; % call the network
        PN_ind = find(sum(spike_time_PN, 2)>0);
        PN_count = length(PN_ind); % how many PN fired
        KC_ind = find(sum(spike_time_KC, 2)>0);
        KC_count = length(KC_ind); % how many KC fired
        KC_fired_total(KC_ind, 2*numTrain + n_train_post) = 1;
        EN_count = sum(sum(spike_time_EN)); % how many spikes EN generated
        PN_activity(2*numTrain + n_train_post) = PN_count;
        KC_activity(2*numTrain + n_train_post) = KC_count;
        EN_activity(2*numTrain + n_train_post) = EN_count;
    end

    % gather information from GPU
    Record.PN_activity = gather(PN_activity);
    Record.KC_activity = gather(KC_activity);
    Record.EN_activity = gather(EN_activity);
    Record.KC_fired_total = gather(KC_fired_total);
    Record.weight_matrix_KC_EN = gather(weight_matrix_KC_EN);
    Record.random_seed = random_seed;
    Record.numKC = numKC;
    save(fullfile(output_dir, sprintf('Record%d_route1.mat', count)),'Record');
    pause(2);
    
    figure(98)
    subplot(311)
    plot(1:3*numTrain, Record.PN_activity, 'gx-')
    subplot(312)
    plot(1:3*numTrain, Record.KC_activity, 'kx-')
    subplot(313)
    plot(1:3*numTrain, Record.EN_activity, 'rx-')
    
    
%-------------------***     3. navigation test     ***--------------------%

    % use pre-defined network parameters (as in section 2)
    % use the weight matrix of KC-EN that stored in 'Record'
    weight_matrix_KC_EN = gpuArray(Record.weight_matrix_KC_EN);
    % load sampled-img positions from the datafile
    img_pos = Raw_images.img_pos; 
    trained_route = img_pos;
  
    record = 0;
    step_size = 0.1; % [m]
    route_length = ceil((size(trained_route,1)-1)*10); % reserve space for errors 
    feeder = trained_route(1,:);% 2D location of the feeder on XY plane
    nest = trained_route(end,:);% 2D location of the nest
    
    %*********************** error measure setup **********************
    perf_measure = zeros(route_length, 1); % recordings for performance
    route_distance = zeros(numPos, 1); % temporal variable
    dis_threshold = 0.2; %[m] threshold for maximum off-route distance
    error_location = []; % record the error position
    record_pos = [];
    %******************************************************************
    
    % Test setup
    
    % '+' to '-' 60 degrees centered in current moving directions
    step_count = 1;scan_range = 120;
    scan_spd = 2; % [degrees]
    % number of images that needed to be test (scanned)
    num_scan_img = scan_range/scan_spd + 1;
    eye_height = 0.01; % [m]
    resolution = 4; % [degrees/pixel]
    fov = 296; % [degrees]
    EN_pool = zeros(1, num_scan_img);
    step_record = zeros(3,route_length);
    
    % Record positions at each step
    current_position = zeros(route_length,2); 
    current_position(1,:) = feeder; % step 1 = Released at the feeder
    
    % Record the moving direction at each step
    moving_direction = [];
    current_pos = 1;
    
    % record for all the EN responses
    EN_response = struct;
    stop_reason = 'step_budget';
    
    % main loop - agent ant moves forward
    for i = 1:route_length,
        step_count  % show the steps
        
        % check if there is a 'moving_direction' or not:
        % e.g., moving_direction for step 1 is empty, because in step 1 
        % the ant has no previous movements and so have to find out by
        % scanning over 120 degrees, centred with the correct heading
        % (image sample heading that we directly give the ant)
        
        % if moving_direction is not empty, that means there is a previous
        % moving direction associated, so the centre will be that direction
        % accordingly.
        
        if isempty(moving_direction), 
            
            display('scan120degree')
            correct_heading = heading(current_pos);% load the stored heading
            KC_activity = gpuArray.zeros(num_scan_img,1);
            EN_activity = gpuArray.zeros(num_scan_img,1);  
            step_record(3,step_count) = 1; % step_record(3,:) indicates naive scan(=1) or not (=0)
            
            for i_scan = 1:num_scan_img,
                temp_pos = current_position(step_count, :);
                % calculate the actual heading
                temp_heading = correct_heading+scan_range/2-(i_scan-1)*scan_spd;
                % grab the corresponding image from the simulated world
                temp_img_1 = ImgGrabber(temp_pos(1), temp_pos(2), eye_height, ...
                    temp_heading, X, Y, Z, colp, fov, resolution);
                temp_img_2 = imresize(temp_img_1, [10, 36]);
                temp_img_3 = 1-double(temp_img_2)/255;
                temp_img_4 = adapthisteq(temp_img_3);
                temp_img_5 = reshape(temp_img_4, numPN, 1);
                % normalisation
                temp_img = temp_img_5./sqrt(sum(temp_img_5.^2));
                temp_img = temp_img*C_I_PN_var;
                % input into gpuArray
                input = gpuArray(temp_img);
                % call the main network file
                GPU_network_fun;
                % main recording on each inputs
                KC_ind = find(sum(spike_time_KC, 2)>0);
                KC_count = length(KC_ind);
                EN_count = sum(sum(spike_time_EN));
                KC_activity(i_scan) = KC_count;
                EN_activity(i_scan) = EN_count;
            end
            
            % Then find the right heading to go forward:
            KC_naive_scan = gather(KC_activity);
            EN_naive_scan = gather(EN_activity);
            EN_response.activity(step_count).scan = EN_naive_scan;
            % find out the minmum value from all the EN responses (the most
            % familiar image from the network's perspective. Then find the 
            % corresponding heading for this value.
            [value, index] = min(EN_naive_scan);
            
            % record heading direction and changing of moving direction 
            % (here the original moving direction is [], so the change is
            % equal to the heading direction
            % step_record(1,:) = decided headings
            step_record(1,step_count) = correct_heading+scan_range/2-(index-1)*scan_spd;
            % step_record(2,:) = EN response
            step_record(2,step_count) = EN_naive_scan(index);
            
            % Moving forward along current heading
            moving_direction = [cosd(step_record(1,step_count)), sind(step_record(1,step_count))];
            current_position(step_count+1,:) = current_position(step_count,:) + step_size*moving_direction;
        else
            previous_heading = step_record(1,step_count-1);
            display('scanning normally');
            step_record(3,step_count) = 1;
            for i_scan = 1:num_scan_img % number of all images
                temp_heading = previous_heading +scan_range/2 - scan_spd*(i_scan-1);
                temp_img_1 = ImgGrabber(current_position(step_count,1),current_position(step_count,2),...
                                        eye_height,temp_heading,X,Y,Z,colp,fov,resolution);
                temp_img_2 = imresize(temp_img_1, [10, 36]);
                temp_img_3 = 1-double(temp_img_2)/255;
                temp_img_4 = adapthisteq(temp_img_3);
                temp_img_5 = reshape(temp_img_4, numPN, 1);
                temp_img = temp_img_5./sqrt(sum(temp_img_5.^2));
                temp_img = temp_img*C_I_PN_var;
                input = gpuArray(temp_img);
                
                % call the network
                GPU_network_fun;
                KC_ind = find(sum(spike_time_KC, 2)>0);
                KC_count = length(KC_ind);
                EN_count = sum(sum(spike_time_EN));
                EN_pool(i_scan) = gather(EN_count);
            end
            % Recording
            EN_response.activity(step_count).scan = EN_pool;
            [step_record(2,step_count), index] = min(EN_pool);
            step_record(1,step_count) = previous_heading + scan_range/2 - scan_spd*(index-1);
            % show result
            moving_direction = [cosd(step_record(1,step_count)), sind(step_record(1,step_count))];
            current_position(step_count+1,:) = current_position(step_count,:) + step_size*moving_direction;
        end
        % if reaches the nest, then stop
        current_distance = norm(nest - current_position(step_count+1,:));
        if current_distance <= dis_threshold
            stop_reason = 'nest_radius';
            break;
        end
        
        % calculate the distance between the current position and all the
        % correct locations -- then find the minimum value (nearest); if it
        % is >= dis_threshold (go off-route and beyond 20cm threshold, have
        % to put the ant back in order to continue, and count as one error)
        
        for i = 1:numPos,
            route_distance(i) = sqrt((current_position(step_count+1,1)-img_pos(i,1))^2 ... 
                                    +(current_position(step_count+1,2)-img_pos(i,2))^2);
        end
        
        % min value:
        [dis_value, ind_pos] = min(route_distance) % show the distance
        record_pos = [record_pos;ind_pos];
        
        % finishing condition
        if ind_pos >= numPos,
            display('finished')
            stop_reason = 'nearest_reference_end';
            break;
        else
            if dis_value > dis_threshold,
                if max(record_pos) < numPos-1,
                    display('above threshold, place back to next nearest point')
                
                    % prevent the ant to be trapped to the same position
                    if ind_pos < max(record_pos),
                        current_position(step_count+2,:) = img_pos(min(max(record_pos)+1,numPos),:);
                        record_pos=[record_pos;max(record_pos)+1];
                        current_pos = max(record_pos);
                    else
                        current_position(step_count+2,:) = img_pos(ind_pos+round(step_size*10),:);
                        record_pos=[record_pos;ind_pos+1];
                        current_pos = ind_pos+1;
                    end
                else
                    display('finished')
                    stop_reason = 'reset_limit_near_end';
                    break;
                end
                % update performance measure
                perf_measure(step_count) = 1;
                error_location = [error_location; current_position(step_count+2,:)];
                step_count = step_count + 2;
                % reset the moving_direction since it's a restart, and the
                % ant is naive about the next location.
                moving_direction = [];
            else
                step_count = step_count + 1;
            end
        end
    end
    % update/crop the current position
    current_position = current_position(1:step_count+1, :);
    step_record = step_record(:, 1:step_count+1);
    % error rate calculation
    error_rate = sum(perf_measure)/step_count
    % for heading
    change_angle = diff(heading);
    
    close all
    figure(97)
    set(figure(1),'Position',[11,51,910,950])
    patch(X',Y',Z')
    hold on
    axis off
    plot(trained_route(:,1),trained_route(:,2),'r','LineWidth',1.0);
    for i = 1:size(error_location,1)
        scatter(error_location(i,1), error_location(i,2), 'g*');
    end
    scatter(feeder(1), feeder(2),'r*');
    scatter(nest(1),nest(2),'r*');
    plot(current_position(1:step_count,1),current_position(1:step_count,2),'bo','LineWidth',0.8, 'MarkerSize', 6);
    pause(2)
    saveas(figure(1), fullfile(output_dir, sprintf('ant%droute1',count)),'fig')
    pause(1)
    saveas(figure(1), fullfile(output_dir, sprintf('ant%droute1',count)), 'jpg')
    
    
    navigation_result = struct;
    navigation_result.step_record = step_record;
    navigation_result.current_position = current_position;
    navigation_result.dis_threshold = dis_threshold;
    navigation_result.error_rate = error_rate;
    navigation_result.weight_matrix = Record.weight_matrix_KC_EN;
    navigation_result.Record = Record;
    navigation_result.EN_response = EN_response;
    navigation_result.error_location = error_location;
    navigation_result.perf_measure = perf_measure;
    navigation_result.change_angle = change_angle;
    navigation_result.stop_reason = stop_reason;
    navigation_result.random_seed = random_seed;
    navigation_result.assisted_resets = sum(perf_measure);
    save(fullfile(output_dir, sprintf('ant%droute1.mat',count)),'navigation_result');
    pause(5);
    
    % finish one ant, update counter
    count = count + 1;
    
end 
