%% Ant navigation models
% random heading 
% paul ardin
% June 2014


close all
clear all

load('AntData.mat');
load('world5000_gray.mat');



for ant = 1:15;
    route_num = 1;
    
    ant_route=strcat('Ant',num2str(ant),'.InwardRouteData.Route',num2str(route_num),'.One_cm_control_points');
    ant_head=strcat('Ant',num2str(ant),'.InwardRouteData.Route',num2str(route_num),'.One_cm_control_points_headings');
    
    % retrieve Ant X Route X positions and heading
    xy=eval(ant_route);
    th=eval(ant_head);
    th=round(th);
    
    % set the x and y co-ordinates
    x=xy(:,1)/100;
    y=xy(:,2)/100;

    % set the size for a simulation step forward
    test_step=0.1;
    
    % set up the plot
    figure(20+ant);
    hold on
    fill(X',Y','k')
    axis equal
    axis off
    plot([x(1),x(end)],[y(1),y(end)],'r*');
    plot(x,y,'r.','MarkerSize',0.5)
    
    x0=x(1);
    y0=y(1);
    th0=th(1);
    pin=1;
    
    % maximum distance the simulation can be rom the route in m
    mx_ro_er=0.2; 
  
   
    perf_measure=[0 0];
    
    
    nimte=floor(size(th,1)/(test_step*100));
    rout_fol=[0 0 0];
    i=1;
    
    clear dl dl2 ddin
    
    ddin=1;
    
    while sqrt((x0-x(end))^2+(y0-y(end))^2) >mx_ro_er;
        
        % get a new random heading
        new_heading=th0+(30*randn);    
        
        % calculate the new position
        th0=(new_heading);
        x0=x0+(test_step*cosd(th0));
        y0=y0+(test_step*sind(th0));
        
        % calculate the distance from the original route
        for k=1:size(x,1);
            xtest=x0-x(k);
            ytest=y0-y(k);
            dl(k)=sqrt(xtest^2+ytest^2);
        end
        
        % and the distance from the previous recapitulated route
        for k=1:size(rout_fol,1)
            xtest2=x0-rout_fol(k,1);
            ytest2=y0-rout_fol(k,2);
            dl2(k)=sqrt(xtest2^2+ytest2^2);
        end
        
        % check whether the new position is too far from the original route or
        % has looped
        if min(dl)>mx_ro_er | min(dl2) < (test_step*0.9);
            % if the error conditions are met pick the index value which is
            % closest to the original route
            [ddum ddin1]=min(dl);
            [ddum ddin2]=min(dl2);
            ddin=max(ddin1,ddin2);
            if ddin<=max(max(perf_measure,[],2));
                ddin=max(max(perf_measure,[],2))+1;
                er=1;
            else
                ddin_old=ddin;
                ddin=ddin+1;
                er=2;
            end
            % scale this to the original route size
            ddin=ddin+(test_step*100);
            if ddin>size(x,1);
                ddin=size(x,1);
            end
            % pick the values from original route using the scaled index
            x0=x(ddin);
            y0=y(ddin);
            th0=th(ddin);
            % record the 'on-route' position and the position in the
            % run that the error occurred
            perf_measure(pin,1:2)=[ddin i];
            % increment the index of the performance measure
            pin=pin+1;
            % draw the position
            figure(20+ant);
            if er==1;
                scatter(x0,y0,'g*');
            else
                scatter(x0,y0,'y*');
            end
            er=0;
        else
            % draw the position
            figure(20+ant);
            scatter(x0,y0,'b');
        end
        % record the route being followed and increment the counter
        rout_fol(i,:)=[x0 y0 th0];
        i=i+1;
        
    end
    er_ran_n(ant,1:2)=[i pin];
end

