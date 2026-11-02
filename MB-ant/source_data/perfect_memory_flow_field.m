%% Ant navigation models
% Paul Ardin 
% June 2014

% perfect memory model
% flow field generator

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

%% set parameters for the visual input

% horizontal field of view
hfov = 296;

% approximate optical resolution 
res = 4;

% resize parameters
sizex = [10 36]; % [10 36];

% generate a test image to determine vector size
it1 = ImgGrabber_fp(5,5,0.01,0,X(1:dens,:),Y(1:dens,:),Z(1:dens,:),colp(1:dens,:),hfov,res);
close all;
it2 = it1(:,1:end-1,2);
it3 = imresize(it2,sizex);
image_vector_size = size(it3,1)*size(it3,2);



%% the image generating loop

% load a specific ant route
ant = 1;
route_num = 1;
ant_route = strcat('Ant',num2str(ant),'.InwardRouteData.Route',num2str(route_num),'.One_cm_control_points');
ant_head = strcat('Ant',num2str(ant),'.InwardRouteData.Route',num2str(route_num),'.One_cm_control_points_headings');
        
% retrieve ant positions and heading
xy = eval(ant_route);
th2 = eval(ant_head);

% length of the ant route during training
train_route_length = floor(size(xy,1)/(train_step*100));


% index points
id_pts = [1:(train_step*100):size(xy,1)];
    
% the simulations use whole number headings so the original headings are
% preferably rounded.
th2 = round(th2);
th = atan2d(xy(id_pts(2:end),2)-xy(id_pts(1:end-1),2),xy(id_pts(2:end),1)-xy(id_pts(1:end-1),1));
th(end+1,1) = atan2d(xy(end,2)-xy(id_pts(end-1),2),xy(end,1)-xy(id_pts(end-1),1));
th = round(th);

% initialise the route vector variable
route_vectors = zeros(train_route_length,image_vector_size);



% run the route generating the images, process them and converting to vectors
for i = 1:train_route_length;    
    % calculate the next position and heading on the route
    img_pos = xy(id_pts(i),:)/100;
    heading = th2(id_pts(i));    % grab an image at that position and heading
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



%% Flowfield generation

head_mod2 = [-60:1:60];

ref_pos = 20; 
img_pos = xy(id_pts(ref_pos),:)/100;
heading = th2(id_pts(ref_pos));
x_stp = 11;
y_stp = 21; 
x_stp_v = floor(x_stp/2)+1;
y_stp_v = floor(y_stp/2)+1;
stp_sz = 5;



% for each point on the x,y grid defined by x_stp and y_stp, generate the
% views across the scan range and find the minimum RMS value
for k = 1:y_stp;    
    for i=1:x_stp;
        for l = 1:121;
            img_pos = xy(id_pts(ref_pos)+((k-y_stp_v)*stp_sz),:)/100;
            heading = th2(id_pts(ref_pos)+((k-y_stp_v)*stp_sz));
            i_1 = ImgGrabber_fp(img_pos(1)+(((i-x_stp_v)*stp_sz)/100),img_pos(2),height,heading+head_mod2(l),X(1:dens,:),Y(1:dens,:),Z(1:dens,:),colp(1:dens,:),hfov,res);
            i_2 = i_1(:,1:end-1,2);
            i_3 = imresize(i_2, sizex);
            i_4 = double(i_3)/255;
            i_5 = 1-i_4;
            i_6 = adapthisteq(i_5);
            i_7 = reshape(i_6, image_vector_size,1);
            i_8 = double(i_7);
            i_9_r = i_8/(sum(i_8.^2))^0.5;
            for j=1:train_route_length;
                i_9 = squeeze(route_vectors(j,:));
                ss_diff_int(j) = rms(i_9_r-i_9');
            end
            ss_diff(k,i,l) = min(ss_diff_int); 
        end        
    end
end


%%  plotting code

% calculate the angles to be plotted

for i=1:y_stp; 
    for j=1:x_stp; 
        [~, idx]=min(ss_diff(i,j,:)); 
        pref_head(i,j)=head_mod2(idx); 
    end; 
end

for i=1:y_stp;
    for j=1:x_stp;
        xy_flow(i,j,1) = xy(id_pts(ref_pos)+((i-y_stp_v)*stp_sz),1)+((j-x_stp_v)*stp_sz);
        xy_flow(i,j,2) = xy(id_pts(ref_pos)+((i-y_stp_v)*stp_sz),2);
    end
end

for i=1:y_stp;
    for j=1:x_stp;
        pref_head_2(i,j) = th2(id_pts(ref_pos)+((i-y_stp_v)*stp_sz)) + pref_head(i,j);
    end
end


% set up and generate the plots

h2 = figure(20);
set(h2,'position',[1     1   847   726])
grasscolormap = zeros(64,3);
grasscolormap(:,2) = linspace(0,1,64);
colormap(grasscolormap);
Xred = and(X(:,1)<7.8,X(:,1)>4.2);
Xsi = sum(Xred);
Xred = repmat(Xred,[1 3]);
Xr = reshape(X(Xred),[Xsi 3]);
Yr = reshape(Y(Xred),[Xsi 3]);
Zr = reshape(Z(Xred),[Xsi 3]);
Cr = reshape(colp(Xred),[Xsi 3]);
world = fill3(Xr',Yr',Zr',Cr');
hold on;
X_surf = [4 8];
Y_surf = [min(Y),max(Y)];
Z_surf = zeros(length(Y_surf), length(X_surf));
Z_surf = Z_surf - 0.01;
surf(X_surf, Y_surf, Z_surf, 'facecolor',[229,183,90]/255, 'edgecolor', 'none')
hold on
axis equal
axis off
view(10,48) 
camzoom(1.3)
set(world, 'edgecolor', 'interp')
x=xy(:,1)/100;
y=xy(:,2)/100;
z_l = size(xy,1);
z_p(1:z_l)=-0.005;
plot3(x,y,z_p,'-r','LineWidth',5)
h3=quiver(xy_flow(:,:,1)/100,xy_flow(:,:,2)/100,cosd(pref_head_2),sind(pref_head_2),0.5,'LineStyle','none')
camva(0.4432)
camtarget([6.2593    5.4649    1.0708])
campos([13.8750  -35.3519   38.3049])
hkid = get(h3,'children');
X_arr = get(hkid(1),'XData');
Y_arr = get(hkid(1),'YData');
for ii = 1:3:length(X_arr)-1 
    ah = annotation('arrow','Color','b','headStyle','plain','HeadLength',15,'LineStyle','-','LineWidth',0.5); 
    set(ah,'parent',gca);
    set(ah,'position',[X_arr(ii) Y_arr(ii) (X_arr(ii+1)-X_arr(ii))*1.1 (Y_arr(ii+1)-Y_arr(ii))*1.1]);
end
for ii = 1:3:length(X_arr)-1
    ah = annotation('arrow','Color','b','headStyle','none','HeadLength',10,'LineStyle','-','LineWidth',2); 
    set(ah,'parent',gca);
    set(ah,'position',[X_arr(ii) Y_arr(ii) (X_arr(ii+1)-X_arr(ii)) (Y_arr(ii+1)-Y_arr(ii))]);
end


