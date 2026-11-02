%% Ant navigation models
% Paul Ardin
% June 2014

% perfect memory with visual compass


%% Initialise the workspace
clear all
close all

tic;

%% load the world and ant data

% world data
load('world5000_gray.mat');

% ant data
load('AntData.mat');

d_st = date;

%% set parameters for the route following

% train step size in meters
train_step = 0.1;

% density of polygons
dens = 5000;

% ant height in meters
height = 0.01;

%% set parameters for the visual input

% horizontal field of view
hfov = 296;

% approximate optical resolution
res = 4;

% resize parameters
sizex = [10 36];

% generate a test image to determine vector size
it1 = ImgGrabber_fp(5,5,0.01,0,X(1:dens,:),Y(1:dens,:),Z(1:dens,:),colp(1:dens,:),hfov,res);
close all;
it2 = it1(:,1:end-1,2);
it3 = imresize(it2,sizex);
image_vector_size = size(it3,1)*size(it3,2);


%% create the perfect memory

ant_range = [1:1:15];
route_nums = 1;

for ant_no = 1:size(ant_range,2);
    
    for route_train = 1:size(route_nums,2);
        route_num = route_nums(route_train);
        
        ant=ant_range(ant_no);
        
        ant_route = strcat('Ant',num2str(ant),'.InwardRouteData.Route',num2str(route_num),'.One_cm_control_points');
        ant_head = strcat('Ant',num2str(ant),'.InwardRouteData.Route',num2str(route_num),'.One_cm_control_points_headings');
        
        % retrieve ant positions and heading
        xy = eval(ant_route);
        th2 = eval(ant_head);
        
        
        
        % length of the ant route during training
        train_route_length = floor(size(xy,1)/(train_step*100));

        
        % index points
        id_pts = [1:(train_step*100):size(xy,1)];
        
        % calculate headings
        th2 = round(th2);
        th = atan2d(xy(id_pts(2:end),2)-xy(id_pts(1:end-1),2),xy(id_pts(2:end),1)-xy(id_pts(1:end-1),1));
        th(end+1,1) = atan2d(xy(end,2)-xy(id_pts(end-1),2),xy(end,1)-xy(id_pts(end-1),1));
        th = round(th);
        
        
        % initialise the route vector variable
        route_vectors = zeros(train_route_length,image_vector_size);
        
        % run the route generating the images, precessing them and converting to vectors
        for i = 1:train_route_length;
            % calculate the next position and heading on the route
            img_pos = xy(id_pts(i),:)/100;
            heading = th2(id_pts(i));    % grab an image at that position and heading
            i_1 = ImgGrabber_fp(img_pos(1),img_pos(2),height,heading,X(1:dens,:),Y(1:dens,:),Z(1:dens,:),colp(1:dens,:),hfov,res);
            % discard red and blue channels (and the last column of the image which
            % is always cyan)
            i_2 = i_1(:,1:end-1,2);
            i_3 = imresize(i_2, sizex);
            i_4 = double(i_3)/255;
            i_5 = 1-i_4;
            i_6 = adapthisteq(i_5);
            i_7 = reshape(i_6, image_vector_size,1);
            i_8 = double(i_7);
            i_9 = i_8/(sum(i_8.^2))^0.5;
            % save the vector
            route_vectors(i,:) = i_9;
            % reset the temporary image variables
            clear i_1 i_2 i_3 i_4 i_5 i_6 i_7 i_8 i_9
        end
        
        
        
        
        %% route recapitulation
        close all
        clear dec dec2 dl dl2 rout_fol
        
        % maximum route error in m
        mx_ro_er = 0.2;
        % test step size in m
        test_step = 0.1;
        
        x=xy(:,1)/100;
        y=xy(:,2)/100;
        
        
        % set up the plot
        figure(20+ant_no);
        hold on
        fill(X',Y','k')
        axis equal
        axis off
        plot([x(1),x(end)],[y(1),y(end)],'r*');
        plot(x,y,'r.','MarkerSize',0.5)
        
        tfig = figure(10);
        set(tfig, 'Position', [100 100 400 400]);
        
        x0=xy(id_pts(1),1)/100;
        y0=xy(id_pts(1),2)/100;
        th0=th(1);
        pin=1;
        
        % calculate the length of the test route
        nimte=floor(size(xy,1)/(test_step*100));
        ddin=1;
        perf_measure=[0 0];
        
        % scan width in degrees
        test_ang=121; 
        test_ang_mid=floor(test_ang/2)+1;
        head_mod = [-(test_ang_mid-1):1:(test_ang_mid-1)];
        
        
        rout_fol=zeros(1,1,3);
        i=1;
        af1 = zeros(test_ang,1);
        
        % main navigation loop
        % keep moving until you get within mx_ro_er of the nest entrance
        while sqrt(((x0-((xy(id_pts(end),1))/100))^2)+((y0-((xy(id_pts(end),2))/100))^2)) > mx_ro_er;
            
            % grab an image at every heading for these co-ordinates
            for j=1:test_ang
                i_1 = ImgGrabber_fp(x0,y0,height,th0+head_mod(j),X(1:dens,:),Y(1:dens,:),Z(1:dens,:),colp(1:dens,:),hfov,res);
                i_2 = i_1(:,1:end-1,2);
                i_3 = imresize(i_2, sizex);
                i_4 = double(i_3)/255;
                i_5 = 1-i_4;
                i_6 = adapthisteq(i_5);
                i_7 = reshape(i_6, image_vector_size,1);
                i_8 = double(i_7);
                i_9_r = i_8/(sum(i_8.^2))^0.5;
                for k=1:train_route_length;
                    i_9 = squeeze(route_vectors(k,:));
                    ss_diff_int(k) = rms(i_9_r-i_9');
                end
                ss_diff(j) = min(ss_diff_int);
                clear i_1 i_2 i_3 i_4 i_5 i_6 i_7 i_8 i_9 i_9_r ss_diff_int
            end
            
            
            % pick the new heading - the minimum RMS difference
            [~, idx_th] = min(ss_diff);
            th0 = th0 + (idx_th - test_ang_mid);
            
            % record the new position on the ant track
            rout_fol(ant_no,i,:)=[x0 y0 th0];
            
            % calculate new x and y
            x0=x0+(test_step*cosd(th0));
            y0=y0+(test_step*sind(th0));
            
            
            clear dl dl2
            
            % calculate the distance of the new point from the actual route
            for k=1:size(id_pts,2);
                xtest=x0-(xy(id_pts(k),1)/100);
                ytest=y0-(xy(id_pts(k),2)/100);
                dl(k)=sqrt(xtest^2+ytest^2);
            end
            
            % calcualte the distance of the new point from the simulation's
            % path
            for k=1:size(rout_fol,1)
                xtest2=x0-rout_fol(k,1);
                ytest2=y0-rout_fol(k,2);
                dl2(k)=sqrt(xtest2^2+ytest2^2);
            end
            
            % if the distance exceeds a threshold value
            % calculate the index position of the minimum distance to the route
            if min(dl)>mx_ro_er || min(dl2) < (test_step*0.9); % the second test is designed to remove looping but it doesn't quite work yet
                [~, ddin]=min(dl);
                % if that index value is at the end, you're at the end
                if ddin==nimte;
                    x0=x0;
                    y0=y0;
                    th0=th0;
                    figure(20+ant_no);
                    scatter(x0,y0,'r*');
                    % otherwise replace the new points with the next position on
                    % the ant route
                else
                    ddin=ddin+1;
                    x0=xy(id_pts(ddin),1)/100;
                    y0=xy(id_pts(ddin),2)/100;
                    th0=th(ddin);
                    % record the 'on-route' position and the position in the
                    % run that the error occurred
                    perf_measure(pin,:)=[ddin i];
                    % increment the index of the performance measure
                    pin=pin+1;
                    figure(20+ant_no);
                    scatter(x0,y0,'g*');
                  end
            else
                % plot the new point 
                figure(20+ant_no);
                scatter(x0,y0,'b');
            end
            
            
            % plot the RMS differences
            figure(10);
            subplot(2,1,1);
            plot(ss_diff);
            subplot(2,1,2);
            plot(af1);
            af1 = ss_diff;
            af_all(i,:) = af1;
            i=i+1;
            
            
            % clear the variables
            clear ss_diff KC_on idx_th
            
            
        end
        
        % record the number of errors
        ant_error(ant_no,route_num,:)=[pin i];
        figure(20+ant)

    end
    
end



