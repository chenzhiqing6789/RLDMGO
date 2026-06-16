function [best_pos, Convergence_curve] = RLDMGO(N, Max_FEs, lb, ub, dim, fobj)
    %INITIALIZATION
    Convergence_curve = [];
    FEs = 0;
    
    current_X = initialization(N, dim, ub, lb);
    current_Fitness = inf * ones(N, 1);
    
    for i = 1:N
        current_Fitness(i, 1) = fobj(current_X(i, :));
        FEs = FEs + 1;
    end
    
    [bestFitness, ~] = min(current_Fitness);
    best_pos = current_X(current_Fitness == bestFitness,:);
    best_pos = best_pos(1,:); % Ensure best_pos is a single row vector
    
    %% --- ENHANCEMENT 1: Reinforcement Learning (RL) Setup ---
    % Action 1: Exploration (DE Operator)
    % Action 2: MGO search operator
    num_actions = 2; 
    Q_table = zeros(N, num_actions); % Q-table for each agent
    learning_rate = 0.5; % Alpha
    epsilon = 0.5; % Initial exploration rate for action selection
    
    %% --- ENHANCEMENT 2: DE Operator Parameter Setup (Non-Adaptive) ---
    memory_size = N;
    M_F = 0.5 * ones(memory_size, 1); % Memory for scaling factor F (now static)
    M_CR = 0.5 * ones(memory_size, 1); % Memory for crossover rate CR (now static)
    archive = []; % Archive for storing improved parent solutions
    archive_size_limit = 2 * N;
    p_best_rate = 0.1; % Percentage for selecting p-best individual
    
    %% --- MGO Algorithm Parameters (for Action 2) ---
    mgo_w = 2;
    mgo_divide_num = dim/4;
    mgo_d1 = 0.2;
     
	iter = 1;
    
    while FEs < Max_FEs
        
        %% --- MGO Pre-calculation for this generation (used if action 2 is chosen) ---
        calPositions = current_X;
        div_num = randperm(dim);
        
        % Divide the population based on the best solution
        for j = 1:max(floor(mgo_divide_num),1)
            th = best_pos(div_num(j));
            index = calPositions(:,div_num(j)) > th;
            if sum(index) < size(calPositions, 1)/2 % Choose the side of the majority
                index = ~index;
            end
            if sum(index) > 0 % Ensure calPositions is not empty
                calPositions = calPositions(index,:);
            end
        end
        
        % Compute the average distance vector D_wind
        D = best_pos - calPositions; 
        D_wind = sum(D, 1) / size(calPositions, 1);
        
        beta = size(calPositions, 1) / N;
        gama = 1 / sqrt(max(1e-10, 1 - power(beta, 2))); % Add small epsilon to avoid division by zero
        
        % Calculate MGO step sizes for this generation
        mgo_step = mgo_w * (rand(1, dim)-0.5) * (1-FEs/Max_FEs);
        mgo_step2 = 0.1 * mgo_w * (rand(1, dim)-0.5) * (1-FEs/Max_FEs) * (1 + 1/2 * (1 + tanh(beta/gama)) * (1-FEs/Max_FEs));
        mgo_step3 = 0.1 * (rand()-0.5) * (1-FEs/Max_FEs);
        mgo_act_input = 1 ./ (1 + (0.5 - 10*(rand(1, dim))));
        mgo_act = actCal(mgo_act_input);
        
        
        for i = 1:N
            %% --- RL: Action Selection (Epsilon-Greedy) ---
            if rand < epsilon
                action = randi(num_actions); % Explore: choose a random action
            else
                [~, action] = max(Q_table(i, :)); % Exploit: choose the best-known action
            end
            
            %% Generate Trial Vector based on the selected action
            trial_X = zeros(1, dim);
            
            if action == 1 % Action 1: Exploration (DE Operator)
                
                % Generate F and CR from static memory
                cr_idx = randi(memory_size);
                CR = normrnd(M_CR(cr_idx), 0.1);
                CR = min(1, max(0, CR));
            
                f_idx = randi(memory_size);
                F = -1;
                while F <= 0 % F must be positive
                    F = cauchy_rnd(M_F(f_idx), 0.1);
                end
                F = min(1, F);

                % Select pbest from top individuals
                p_best_count = max(2, ceil(p_best_rate * N));
                [~, sorted_indices] = sort(current_Fitness);
                pbest_idx = sorted_indices(randi(p_best_count));
                p_best_pos = current_X(pbest_idx, :);

                % Select r1 from population and r2 from archive
                r1_idx = randi(N); 
                while r1_idx == i, r1_idx = randi(N); end
                
                % Combine current population and archive for r2 selection
                pop_archive = [current_X; archive];
                r2_idx = randi(size(pop_archive, 1));
                while r2_idx == i, r2_idx = randi(size(pop_archive, 1)); end
                
                % Mutation: DE/current-to-pbest/1
                v = current_X(i, :) + F * (p_best_pos - current_X(i, :)) + F * (current_X(r1_idx, :) - pop_archive(r2_idx, :));

                % Binomial Crossover
                j_rand = randi(dim);
                for j = 1:dim
                    if rand < CR || j == j_rand
                        trial_X(j) = v(j);
                    else
                        trial_X(j) = current_X(i, j);
                    end
                end

            elseif action == 2 % Action 2: MGO search operator
                trial_X = current_X(i, :);
                
                % Spore dispersal search (from MGO)
                if rand() > mgo_d1
                    trial_X = trial_X + mgo_step .* D_wind; 
                else
                    trial_X = trial_X + mgo_step2 .* D_wind;
                end
                
                % Dual propagation search (from MGO)
                if rand() < 0.8
                    if rand() > 0.5
                        trial_X(div_num(1)) = best_pos(div_num(1)) + mgo_step3 * D_wind(div_num(1));
                    else
                        trial_X = (1 - mgo_act) .* trial_X + mgo_act .* best_pos;
                    end
                end
            end
            
            %% Boundary control
            trial_X = BoundaryControl(trial_X, lb, ub);
            
            %% Greedy Selection & Updates
            trial_Fitness = fobj(trial_X);
            FEs = FEs + 1;
            
            if trial_Fitness < current_Fitness(i)
                % Success: update position and fitness
                old_solution = current_X(i, :);
                current_X(i, :) = trial_X;
                current_Fitness(i) = trial_Fitness;
                reward = 1; % Positive reward for improvement
                
                % Add the replaced parent to the archive (if DE action was used)
                if action == 1
                    archive(end+1, :) = old_solution;
                end
                
                % Update global best if necessary
                if trial_Fitness < bestFitness
                    bestFitness = trial_Fitness;
                    best_pos = trial_X;
                end
            else
                % Failure
                reward = 0; % No reward for no improvement
            end
            
            %% RL: Update Q-table
            Q_table(i, action) = (1 - learning_rate) * Q_table(i, action) + learning_rate * reward;
        end
        
        %% Trim archive if it exceeds the size limit
        if size(archive, 1) > archive_size_limit
            rand_indices = randperm(size(archive, 1));
            archive = archive(rand_indices(1:archive_size_limit), :);
        end
        
        Convergence_curve(iter) = bestFitness;
        iter = iter + 1;
        epsilon = epsilon * 0.99; % Decay epsilon over time
    end
end  

%% --- Helper Functions ---

function X = initialization(N, dim, ub, lb)
    Boundary_no = size(ub, 2); % Number of boundaries
    if Boundary_no == 1
        X = rand(N, dim) .* (ub - lb) + lb;
    else
        X = zeros(N, dim);
        for i = 1:dim
            ub_i = ub(i);
            lb_i = lb(i);
            X(:,i) = rand(N,1).*(ub_i-lb_i)+lb_i;
        end
    end
end

function X = BoundaryControl(X, low, up)
    if numel(low) == 1 % Single boundary for all dimensions
        X(X < low) = low;
        X(X > up) = up;
    else % Boundary for each dimension
        X = max(X, low);
        X = min(X, up);
    end
end
    
function r = cauchy_rnd(mu, gamma)
    % Generates a random number from a Cauchy distribution
    r = mu + gamma * tan(pi * (rand - 0.5));
end

% --- Helper function from MGO ---
function [act] = actCal(X)
    act = X;
    act(act>=0.5) = 1;
    act(act<0.5) = 0;
end