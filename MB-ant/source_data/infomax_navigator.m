%% Ant navigation models
% Infomax
% Paul Ardin
% June 2014
% amended for Infomax February 2015


% NB  this version adjusts the route angles on the basis of step size

%% Initialise the workspace
clear all
close all



%% load the world and ant data

% world data
load('world5000_gray.mat');

% ant data
load('AntData.mat');

%% set parameters for the route following

% train step size in meters
train_step = 0.1;

% density of polygons
dens = 5000;

% ant height in meters
height = 0.01;

% number of training runs for weights
num_trains=1;

% the learning rate
mu=1.1;

%% set parameters for the visual input

% horizontal field of view
hfov = 296;

% approximate optical resolution
res = 4;

% resize parameters
sizex =  [10 36];

% generate a test image to determine vector size
it1 = ImgGrabber_fp(5,5,0.01,0,X(1:dens,:),Y(1:dens,:),Z(1:dens,:),colp(1:dens,:),hfov,res);
close all;
it2 = it1(:,1:end-1,2);
it3 = imresize(it2,sizex);
image_vector_size = size(it3,1)*size(it3,2);

%% generate the infomax net

% the infomax network is the same size as the image along the input node
% dimension
if image_vector_size<2000;
    nhid=image_vector_size; % hidden nodes symmetrical with input size
else
    nhid=1000; % for large size inputs e.g. 1 degree resolution reduce the hidden layer to avoid memory overload
end


%% the image generating loop

% load a specific ant route

ant_range = [1:1:1];
route_num = 1;

for ant_no = 1:size(ant_range,2);
    
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
    
    % initialise the weights for the infomax net
    if route_num==1;
        W = 0 + 1.*randn(image_vector_size,nhid);
        W=W';
    end
    
    N=size(id_pts,2);
    P=image_vector_size;
    [H,V] = size(W);
    
    % the simulations use whole number headings so the original headings are
    % preferably rounded.
    th2 = round(th2);
    th = atan2d(xy(id_pts(2:end),2)-xy(id_pts(1:end-1),2),xy(id_pts(2:end),1)-xy(id_pts(1:end-1),1));
    th(end+1,1) = atan2d(xy(end,2)-xy(id_pts(end-1),2),xy(end,1)-xy(id_pts(end-1),1));
    th = round(th);
    
    % initialise the route vector variable
    route_vectors = zeros(train_route_length,image_vector_size);
    
    % run the route generating the images and converting to binary vectors
    for i = 1:train_route_length;
        img_pos = xy(id_pts(i),:)/100;
        heading = th(i);
        % grab an image at that position and heading
        i_1 = ImgGrabber_fp(img_pos(1),img_pos(2),height,heading,X(1:dens,:),Y(1:dens,:),Z(1:dens,:),colp(1:dens,:),hfov,res);
        % discard red and blue channels 
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
 
    close all
    
    
    
    %% route learning loop

    % run the infomax learnign rule against every image vector
    for j = 1:num_trains;
        for i = 1:train_route_length;
            training_image = route_vectors(i,:);
            u=W*training_image';
            g=tanh(u);
            W = W+ mu/P * (eye(H)-(g+u)*u') * W;
            if isnan(W)
                fprintf('weights error\n')
                break
            end
            
        end
        w_store(ant,j,:,:)=W;
    end

    close all

    
    %% the route recapitulation loop

    
    close all
    clear dec dec2 dl dl2 rout_fol
    
    % maximum allowed deviation from the route in m
    mx_ro_er = 0.2; 
    % step size for route recapitulation in m
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
    
    % determine the size of the route
    nimte=floor(size(xy,1)/(test_step*100));
    ddin=1;
    perf_measure=[0 0];
    % set the scan angle size
    test_ang=121; 
    test_ang_mid=floor(test_ang/2)+1;
    head_mod = [-(test_ang_mid-1):1:(test_ang_mid-1)];
    rout_fol=zeros(1,1,3);
    i=1;
    
    af1 = zeros(test_ang,1);
    
    % is the simulation is not within mx_ro_er of the nest keep running
    while sqrt(((x0-((xy(id_pts(end),1))/100))^2)+((y0-((xy(id_pts(end),2))/100))^2)) > mx_ro_er;
        
        % generate an image at every heading for the current co-ordinates
        for j=1:test_ang
            i_1 = ImgGrabber_fp(x0,y0,height,th0+head_mod(j),X(1:dens,:),Y(1:dens,:),Z(1:dens,:),colp(1:dens,:),hfov,res);
            % discard red and blue channels 
            i_2 = i_1(:,1:end-1,2);
            i_3 = imresize(i_2, sizex);
            i_4 = double(i_3)/255;
            i_5 = 1-i_4;
            i_6 = adapthisteq(i_5);
            i_7 = reshape(i_6, image_vector_size,1);
            i_8 = double(i_7);
            i_9 = i_8/(sum(i_8.^2))^0.5;
            % save the vector
            ang_vectors(i,j,:) = i_9;
            % calculate the Infomax value
            result=W*i_9;
            ang_fami(j) = sum(abs(result));
            % reset the temporary image variables
            clear i_1 i_2 i_3 i_4 i_5 i_6 i_7 i_8 i_9
        end
        
        % pick the new heading, which is the minimum Infomax value
        [~, idx_th] = min(ang_fami);
        th0 = th0 + (idx_th - test_ang_mid);
        
        % record the new point 
        rout_fol(ant_no,i,:)=[x0 y0 th0];
        
        % calculate new x and y
        x0=x0+(test_step*cosd(th0));
        y0=y0+(test_step*sind(th0));
        
        
        clear dl dl2
        
        % calculate the distance of the new point from the actual ant route
        for k=1:size(id_pts,2);
            xtest=x0-(xy(id_pts(k),1)/100);
            ytest=y0-(xy(id_pts(k),2)/100);
            dl(k)=sqrt(xtest^2+ytest^2);
        end
        
        % calculate the distance of the new point from the simulation's
        % recapitulated route
        for k=1:size(rout_fol,1)
            xtest2=x0-rout_fol(k,1);
            ytest2=y0-rout_fol(k,2);
            dl2(k)=sqrt(xtest2^2+ytest2^2);
        end
        
        % if the distance exceeds mx_ro_er or is close to the recapitulated
        % route (e.g. looping)
        % calculate the index position of the minimum distance to the route
        if min(dl)>mx_ro_er || min(dl2) < (test_step*0.9); 
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
                perf_measure(pin,:)=[ddin i];
                pin=pin+1;
                figure(20+ant_no);
                scatter(x0,y0,'g*');
            end
        else
            figure(20+ant_no);
            scatter(x0,y0,'b');
        end
         
        
        % plot the Infomax values
        figure(10);
        subplot(2,1,1);
        plot(ang_fami);
        subplot(2,1,2);
        plot(af1);
        af1 = ang_fami;
        af_all(i,:) = af1;
        i=i+1;
 
        
        % clear the variables
        clear ang_fami KC_on idx_th
        
        
    end
    d_st = date;
    figure(20+ant)
 
end












