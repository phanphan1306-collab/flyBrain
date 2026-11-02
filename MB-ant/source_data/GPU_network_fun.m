% GPU_network_fun.m
% this script uses pre-defined network parameters to setup Izhikevich
% neurons and conductance-based synapses, runs and records the network
% activities.

% the output will be PN/KC/EN activities in each layer
% weight_matrix_KC_EN will be modified by learning, if there is BA presented;
% otherwise the weight_matrix will not be changed.


% Reward signal
BA = 0;

% visualise the process: recordings of voltage KC, synapses PN
% synapses_record = gpuArray.zeros(numPN, interval/dt);
% voltage_KC = gpuArray.zeros(numKC, interval/dt);
% KC_ind = 590; % manually choose one fired KC

% synapses
synapses_PN_KC = gpuArray.zeros(numPN, numKC);
synapses_KC_EN = gpuArray.zeros(numKC, numEN);
% spikes
spike_PN = gpuArray.zeros(numPN,1);
spike_KC = gpuArray.zeros(numKC,1);
spike_EN = gpuArray.zeros(numEN,1);
% input current
I_PN = gpuArray.zeros(numPN, 1);
I_KC = gpuArray.zeros(numKC, 1);
I_EN = gpuArray.zeros(numEN, 1);

% Izhikevich neuron membrane voltage and initial status
PN = [gpuArray.ones(numPN,1), gpuArray.zeros(numPN,1)]*(-60);
KC = [gpuArray.ones(numKC,1), gpuArray.zeros(numKC,1)]*(-85);
EN = gpuArray([1, 0])*(-60);
% snaptic tage for eligibility trace
synaptic_tag_KC_EN = gpuArray.zeros(numKC, numEN);
% biogenic amine concentration
concentration_BA_KC_EN = gpuArray.zeros(numKC, numEN);
% pre- and post- spiking time recorders
t_spike_KC = gpuArray.ones(numKC,1)*(-10000);
t_spike_EN = gpuArray.ones(numEN,1)*(-10000);
% general spike Recordings:
spike_time_PN = gpuArray.zeros(numPN, interval/dt);
spike_time_KC = gpuArray.zeros(numKC, interval/dt);
spike_time_EN = gpuArray.zeros(numEN, interval/dt);
pre_post_spike_occured = gpuArray.zeros(numKC, numEN);
% time difference
delta_t = gpuArray.zeros(numKC, numEN);
PN_spikes = gpuArray.zeros(numPN, numKC);
% KC_spikes = gpuArray.zeros(numKC, numEN);

%% Main loop - iteration over time interval/dt

for idt = 1 : interval/dt
    t = idt*dt;
    BA = 0;
    if t < 40.01
        I_PN = input;
        if t == 40.0 && reward == 1
            BA = 0.5; % give reward at the end of the CS onset
        end
    elseif t <= interval
        I_PN = null_input; % null input for a short time after CS offset
    end

    % update PN_KC synapses
    PN_spikes = bsxfun(@times, connection_PN_KC, spike_PN);
    synapses_PN_KC = arrayfun(@PN_KC_synapse, dt, PN_spikes, synapses_PN_KC);
    % Recordings:
    % synapses_record(:,idt) = synapses_PN_KC(:,KC_ind);
    
    % update KC_EN synapses
    pre_post_spike_occured = bsxfun(@max, spike_KC, spike_EN);
    delta_t = t_spike_KC-t_spike_EN;
    [synapses_KC_EN, weight_matrix_KC_EN, synaptic_tag_KC_EN, concentration_BA_KC_EN] = arrayfun(@KC_EN_synapse, dt, spike_KC, synapses_KC_EN,...
                                    weight_matrix_KC_EN, synaptic_tag_KC_EN, delta_t, pre_post_spike_occured, concentration_BA_KC_EN, BA);
    I_KC = sum(g_PN_KC*bsxfun(@times, synapses_PN_KC, (0-KC(:,1))'))';
    I_EN = weight_matrix_KC_EN'*synapses_KC_EN*(0-EN(1));
    
    % update PN neurons
    [spike_PN, PN(:,1), PN(:,2)] = arrayfun(@PN_neuron, dt, PN(:,1), PN(:,2), I_PN);
    spike_time_PN(:,idt) = spike_PN;

    % update KC neurons
    [spike_KC, t_spike_KC, KC(:,1), KC(:,2)] = arrayfun(@KC_neuron, dt, t, KC(:,1), KC(:,2), I_KC, t_spike_KC);
    spike_time_KC(:,idt) = spike_KC;
    % Recordings
    % voltage_KC(:,idt) = KC(:,1);
    
    % update EN neurons
    [spike_EN, t_spike_EN, EN(1), EN(2)] = arrayfun(@EN_neuron, dt, t, EN(1), EN(2), I_EN, t_spike_EN);
    spike_time_EN(:,idt) = spike_EN;
end
% % plot
% figure(11)
% PN_ind = find(connection_PN_KC(:,KC_ind) == 1);
% PN_index = gather(PN_ind);
% raster_PN_KC = synapses_record(PN_ind, :);
% 
% raster_PN_KC_plot = gather(raster_PN_KC);
% KC_plot = gather(voltage_KC(KC_ind,:));
% for i = 1:14 % loop through all the 14 connections PN2KC
%     subplot(16, 1, i)
%     plot(1:interval/dt, raster_PN_KC_plot(i,:), 'b');
%     ylabel(sprintf('#%d', PN_index(i)))
% end
% subplot(818)
% plot(1:interval/dt, KC_plot, 'r');
% ylabel(sprintf('#dKC', KC_ind))
% xlabel('time[ms]')
% 
% % raster plot
% figure(12)
% subplot(311)
% raster_PN = repmat([1:numPN]', 1, interval/dt).*gather(spike_time_PN);
% plot(1:interval/dt, raster_PN, 'k.', 'MarkerSize', 0.1);
% subplot(312)
% raster_KC = repmat([1:numKC]', 1, interval/dt).*gather(spike_time_KC);
% plot(1:interval/dt, raster_KC, 'b.', 'MarkerSize', 0.1);
% subplot(313)
% raster_EN = repmat([1:numEN]', 1, interval/dt).*gather(spike_time_EN);
% plot(1:interval/dt, raster_EN, 'rx', 'MarkerSize', 1.0);
% axis([0 interval/dt+10, 0 3])
