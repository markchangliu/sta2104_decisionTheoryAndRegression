using Distributions
using LinearAlgebra
using Test
using Plots
using Zygote: gradient
using Logging

# Q1(1)
losses = [[10, 0],
          [1, 50],
          [0, 200]];

num_actions = length(losses);


function expected_loss_of_action(prob_spam, action)
    # Return expected loss over a Bernoulli random variable
    # with mean prob spam.
    # Losses are given by the table above.
    expected_loss=losses[action][1].*prob_spam
    expected_loss=expected_loss+losses[action][2].*(1 .- prob_spam)
    return expected_loss
end

prob_range = range(0., stop=1., length=500);

for action in 1:num_actions
  display(plot!(prob_range, expected_loss_of_action(prob_range, action)))
end
# savefig("Q1(1)")

#Q1(2)
function optimal_action(prob_spam)
  # return best action given the probability of spam.
  # Hint: Julia's findmin function might be helpful.
  l1=expected_loss_of_action(prob_spam, 1)
  l2=expected_loss_of_action(prob_spam, 2)
  l3=expected_loss_of_action(prob_spam, 3)
  l=[l1 l2 l3]
  best_loss, index=findmin(l, dims=2)
  best_action=getindex.(index, [1 2])[:, 2]
  return (best_loss, best_action)
end

#Q1(3)
prob_range = range(0, stop=1., length=500);
optimal_losses = [];
optimal_actions = [];
optimal_losses, optimal_actions=optimal_action(prob_range);
plot(prob_range, optimal_losses, linecolor=optimal_actions, legend=false)
savefig("Q1(3)")

#Q2.2
function target_f1(x, σ_true=0.3)
    noise = randn(size(x))
    y = 2x .+ σ_true.*noise
    return vec(y)
end;

function target_f2(x)
    noise = randn(size(x))
    y = 2x + norm.(x)*0.3.*noise
    return vec(y)
end;

function target_f3(x)
    noise = randn(size(x))
    y = 2x + 5sin.(0.5*x) + norm.(x)*0.3.*noise
    return vec(y)
end

#Q2.2(1)
function sample_batch(target_f, batch_size)
    x=rand(Uniform(0, 20), batch_size)
    x=reshape(x,:,batch_size)
    y=target_f(x)
    return (x, y)
end;

@testset "sample dimensions are correct" begin
  m = 1 # dimensionality
  n = 200 # batch-size
  for target_f in (target_f1, target_f2, target_f3)
    x,y = sample_batch(target_f,n)
    @test size(x) == (m,n)
    @test size(y) == (n,)
  end
end

#Q2.2(2)
x1, y1=sample_batch(target_f1, 1000);
plot_f1=plot(vec(x1), y1, seriestype=:scatter, markersize=1, legend=false)
savefig("Q2_2(2)_target_f1")

x2, y2=sample_batch(target_f2, 1000);
plot_f2=plot(vec(x2), y2, seriestype=:scatter, markersize=1, legend=false)
savefig("Q2_2(2)_target_f2")

x3, y3=sample_batch(target_f3, 1000);
plot_f3=plot(vec(x3), y3, seriestype=:scatter, markersize=1, legend=false)
savefig("Q2_2(2)_target_f3")

#Q2.3(1)
function beta_mle(X, y)
    beta=inv(X*X')*X*y
    return beta
end;

n=1000;

#Q2.3(2)
x1, y1=sample_batch(target_f1, 1000);
beta_mle_f1=beta_mle(x1, y1);

x2, y2=sample_batch(target_f2, 1000);
beta_mle_f2=beta_mle(x2, y2);

x3, y3=sample_batch(target_f3, 1000);
beta_mle_f3=beta_mle(x3, y3);

x1_sort=sort(x1, dims=2);
y1_hat=x1_sort'*beta_mle_f1;

x2_sort=sort(x2, dims=2);
y2_hat=x2_sort'*beta_mle_f2;

x3_sort=sort(x3, dims=2);
y3_hat=x3_sort'*beta_mle_f3;

#Q2.3(3)

plot!(plot_f1, vec(x1_sort), x1_sort'*beta_mle_f1,
    ribbon=1, fillalpha=.5, linecolor=:red,
    fillcolor=:red, thickness_scaling=2)
savefig("Q2_3(3)_target_f1")

plot!(plot_f2, vec(x2_sort), x2_sort'*beta_mle_f2,
    ribbon=1, fillalpha=.5, linecolor=:red,
    fillcolor=:red, thickness_scaling=2)
savefig("Q2_3(3)_target_f2")

plot!(plot_f3, vec(x3_sort), x3_sort'*beta_mle_f3,
    ribbon=1, fillalpha=.5, linecolor=:red,
    fillcolor=:red, thickness_scaling=2)
savefig("Q2_3(3)_target_f3")

#Q2.4(1)
function gaussian_log_likelihood(μ, σ, x)
  """
  compute log-likelihood of x under N(μ,σ)
  """
  log_prob1=-log.(σ .* sqrt(2*pi))
  log_prob2=-0.5 .* ((x-μ)./σ).^2
  log_prob=log_prob1 .+ log_prob2
  return log_prob
end

@testset "Gaussian log likelihood" begin
    using Distributions: pdf, Normal, logpdf
    # Scalar mean and variance
    x = randn()
    μ = randn()
    σ = rand()
    @test size(gaussian_log_likelihood(μ,σ,x)) == () # Scalar log-likelihood
    @test gaussian_log_likelihood.(μ,σ,x) ≈ logpdf.(Normal(μ,σ),x) # Correct Value
    # Vector valued x under constant mean and variance
    x = randn(100)
    μ = randn()
    σ = rand()
    @test size(gaussian_log_likelihood.(μ,σ,x)) == (100,) # Vector of log-likelihoods
    @test gaussian_log_likelihood.(μ,σ,x) ≈ logpdf.(Normal(μ,σ),x) # Correct Values
    # Vector valued x under vector valued mean and variance
    x = randn(10)
    μ = randn(10)
    σ = rand(10)
    @test size(gaussian_log_likelihood.(μ,σ,x)) == (10,) # Vector of log-likelihoods
    @test gaussian_log_likelihood.(μ,σ,x) ≈ logpdf.(Normal.(μ,σ),x) # Correct Values
end

#Q2.4(2)
function lr_model_nll(β,x,y;σ=1.)
    mean=vec(x'*β)
    return -sum(gaussian_log_likelihood(mean, σ, y))
end

#Q2.4(3)
for n in (10,100,1000)
    println("--------  $n  ------------")
    for target_f in (target_f1,target_f2, target_f3)
      println("--------  $target_f  ------------")
      for σ_model in (0.1,0.3,1.,2.)
        println("--------  $σ_model  ------------")
        x,y = sample_batch(target_f,n)
        β_mle = beta_mle(x, y)
        nll = lr_model_nll(β_mle,x,y;σ=σ_model)
        println("Negative Log-Likelihood: $nll")
      end
    end
end

#Q2.5(1)
@testset "Gradients wrt parameter" begin
    β_test = randn()
    σ_test = rand()
    x,y = sample_batch(target_f1,100)
    f=beta -> lr_model_nll(beta, x, y; σ=σ_test)
    ad_grad =gradient(f, β_test)
    hand_derivative = ((-y'*x' .+ β_test'*x*x')./(σ_test.^2))[1]
    @test ad_grad[1] ≈ hand_derivative
end

#Q2.5.1(1)
# Print training progress to REPL, not pdf
function train_lin_reg(target_f, β_init; bs= 100, lr = 1e-6, iters=1000, σ_model = 1. )
    β_curr = β_init
    for i in 1:iters
      x,y = sample_batch(target_f, bs)
      loss=lr_model_nll(β_curr, x, y; σ=σ_model)
      #@info "loss: $(loss)  β: $β_curr" #log loss, if you want to monitor training progress
      f=beta -> lr_model_nll(beta, x, y; σ=σ_model)
      grad_β = gradient(f, β_curr)[1] #compute gradients
      β_curr = β_curr .- lr .* grad_β #gradient descent
    end
    return β_curr
end;

#Q2.5.1(2)
β_init = 1000*randn();# Initial parameter

β_learned_f1= train_lin_reg(target_f1, β_init); #call training function
x_f1,y_f1 = sample_batch(target_f1, 1000);
x_f1_sort=sort(x_f1, dims=2);
y_f1_hat=x_f1_sort'*β_learned_f1;
plot_f1=plot(vec(x_f1), y_f1, seriestype=:scatter, markersize=1)
plot!(plot_f1, vec(x_f1_sort), x_f1_sort'*β_learned_f1,
    ribbon=1, fillalpha=.5, linecolor=:red,
    fillcolor=:red, thickness_scaling=2)
savefig("Q2_5_1(2)_target_f1")

β_learned_f2= train_lin_reg(target_f2, β_init);
x_f2,y_f2 = sample_batch(target_f2, 1000);
x_f2_sort=sort(x_f2, dims=2);
y_f2_hat=x_f2_sort'*β_learned_f2;
plot_f2=plot(vec(x_f2), y_f2, seriestype=:scatter, markersize=1)
plot!(plot_f2, vec(x_f2_sort), x_f2_sort'*β_learned_f2,
    ribbon=1, fillalpha=.5, linecolor=:red,
    fillcolor=:red, thickness_scaling=2)
savefig("Q2_5_1(2)_target_f2")

β_learned_f3= train_lin_reg(target_f3, β_init);
x_f3,y_f3 = sample_batch(target_f3, 1000);
x_f3_sort=sort(x_f3, dims=2);
y_f3_hat=x_f3_sort'*β_learned_f3;
plot_f3=plot(vec(x_f3), y_f3, seriestype=:scatter, markersize=1)
plot!(plot_f3, vec(x_f3_sort), x_f3_sort'*β_learned_f3,
    ribbon=1, fillalpha=.5, linecolor=:red,
    fillcolor=:red, thickness_scaling=2)
savefig("Q2_5_1(2)_target_f3")

#Q2.5.2(1)
# Neural Network Function
# size(X)=D,B
# θ is a tuple(weights, bias ...) containing all weights and bias
function neural_net(x,θ)
    θ1=θ[1] # size(θ1)=H,D
    b1=θ[2] # size(b1)=H,1
    θ2=θ[3] # size(θ1)=D,H
    b2=θ[4] # size(b1)=D,1
    z=θ1*x + b1*ones(1, size(x)[2]) # size(z)=H,B
    h=tanh.(z) # size(h)=H,B
    y=θ2*h + b2*ones(1, size(x)[2]) # size(h)=D,B
    return vec(y)
end;

# Random initial Parameters
θ = (randn((10, 1)), randn((10, 1)), randn((1, 10)), randn((1,1)));

@testset "neural net mean vector output" begin
    n = 100
    x,y = sample_batch(target_f1,n)
    μ = neural_net(x,θ)
    @test size(μ) == (n,)
end

#Q2.5.2(2)
function nn_model_nll(θ,x,y;σ=1)
    mean_nn=neural_net(x,θ)
    return -sum(gaussian_log_likelihood(mean_nn, σ, y))
end

#Q2.5.2(3)
function train_nn_reg(target_f, θ_init; bs= 100, lr = 1e-5, iters=1000, σ_model = 1. )
    θ_curr = θ_init
    for i in 1:iters
      x,y = sample_batch(target_f, bs)
      loss=nn_model_nll(θ_curr,x,y;σ=σ_model)
      #@info "loss: $(loss)" #log loss, if you want to montior training
      f_θ = θ -> nn_model_nll(θ,x,y;σ=σ_model)
      grad_θ = gradient(f_θ, θ_curr)[1] #compute gradients
      θ1=θ_curr[1] - lr .* grad_θ[1]
      b1=θ_curr[2] - lr .* grad_θ[2]
      θ2=θ_curr[3] - lr .* grad_θ[3]
      b2=θ_curr[4] - lr .* grad_θ[4]
      θ_curr = (θ1, b1, θ2, b2) #gradient descent
    end
    return θ_curr
end

#Q2.5.2(4)
θ_init = (randn((10, 1)), randn((10, 1)), randn((1, 10)), randn((1,1)));

θ_learned_f1 =train_nn_reg(target_f1, θ_init; iters=5000);
x_f1,y_f1 = sample_batch(target_f1, 1000);
x_f1_sort=sort(x_f1, dims=2);
y_f1_hat=neural_net(x_f1_sort,θ_learned_f1);
plot_f1=plot(vec(x_f1), y_f1, seriestype=:scatter, markersize = 1, legend=false)
plot!(plot_f1, vec(x_f1_sort), y_f1_hat, ribbon=1, fillalpha=.5, linecolor=:red,
    fillcolor=:red, thickness_scaling=2)
savefig("Q2_5_2(4)_target_f1")

θ_learned_f2 =train_nn_reg(target_f2, θ_init; iters=5000);
x_f2,y_f2 = sample_batch(target_f2, 1000);
x_f2_sort=sort(x_f2, dims=2);
y_f2_hat=neural_net(x_f2_sort,θ_learned_f2);
plot_f2=plot(vec(x_f2), y_f2, seriestype=:scatter, markersize = 1, legend=false)
plot!(plot_f2, vec(x_f2_sort), y_f2_hat, ribbon=1, fillalpha=.5, linecolor=:red,
    fillcolor=:red, thickness_scaling=2)
savefig("Q2_5_2(4)_target_f2")

θ_learned_f3 =train_nn_reg(target_f3, θ_init; iters=5000);
x_f3,y_f3 = sample_batch(target_f3, 1000);
x_f3_sort=sort(x_f3, dims=2);
y_f3_hat=neural_net(x_f3_sort,θ_learned_f3);
plot_f3=plot(vec(x_f3), y_f3, seriestype=:scatter, markersize = 1, legend=false)
plot!(plot_f3, vec(x_f3_sort), y_f3_hat, ribbon=1, fillalpha=.5, linecolor=:red,
    fillcolor=:red, thickness_scaling=2)
savefig("Q2_5_2(4)_target_f3")

# Q2.5.3(1)
function neural_net_w_var(x,θ)
    θ1=θ[1] # size(θ1)=H,D
    b1=θ[2] # size(b1)=H,1
    θ2=θ[3] # size(θ1)=D,H
    b2=θ[4] # size(b1)=D,1
    z=θ1*x + b1*ones(1, size(x)[2]) # size(z)=H,B
    h=tanh.(z) # size(h)=H,B
    y=θ2*h + b2*ones(1, size(x)[2]) # size(h)=D,B
    return (y[1,:], y[2,:])
end;

# Random initial Parameters
θ=(randn((10, 1)), randn((10, 1)), randn((2, 10)), randn((2,1)));

@testset "neural net mean and logsigma vector output" begin
    n = 100
    x,y = sample_batch(target_f1,n)
    μ, logσ = neural_net_w_var(x,θ)
    @test size(μ) == (n,)
    @test size(logσ) == (n,)
end

# Q2.5.3(2)
function nn_with_var_model_nll(θ,x,y)
    μ, logσ = neural_net_w_var(x,θ)
    loss=-sum(gaussian_log_likelihood(μ, exp.(logσ), y))
    return loss
end;

# Q2.5.3(3)
function train_nn_w_var_reg(target_f, θ_init; bs= 100, lr = 1e-4, iters=10000)
    θ_curr = θ_init
    for i in 1:iters
      x,y = sample_batch(target_f, bs)
      loss=nn_with_var_model_nll(θ_curr,x,y)
      @info "loss: $(loss)"
      f_θ = θ -> nn_with_var_model_nll(θ,x,y)
      grad_θ = gradient(f_θ, θ_curr)[1] #compute gradients
      θ1=θ_curr[1] - lr .* grad_θ[1]
      b1=θ_curr[2] - lr .* grad_θ[2]
      θ2=θ_curr[3] - lr .* grad_θ[3]
      b2=θ_curr[4] - lr .* grad_θ[4]
      θ_curr = (θ1, b1, θ2, b2) #gradient descent
    end
    return θ_curr
end;

# Q2.5.3(4)
θ_init = (randn((10, 1)), randn((10, 1)), randn((2, 10)), randn((2,1)));

θ_learned_f1 =train_nn_w_var_reg(target_f1, θ_init; lr = 1e-5, iters=100000);
x_f1,y_f1 = sample_batch(target_f1, 1000);
x_f1_sort=sort(x_f1, dims=2);
μ_f1_hat, logσ_f1=neural_net_w_var(x_f1_sort,θ_learned_f1);
plot_f1=plot(vec(x_f1), y_f1, seriestype=:scatter, markersize = 1, legend=false)
plot!(plot_f1, vec(x_f1_sort), μ_f1_hat, ribbon=vec(exp.(logσ_f1)), fillalpha=.5, linecolor=:red,
    fillcolor=:red, thickness_scaling=2, legend=false)
savefig("Q2_5_3(4)_f1")

θ_learned_f2 =train_nn_w_var_reg(target_f2, θ_init; lr = 1e-5, iters=100000);
x_f2,y_f2 = sample_batch(target_f2, 1000);
x_f2_sort=sort(x_f2, dims=2);
μ_f2_hat, logσ_f2=neural_net_w_var(x_f2_sort,θ_learned_f2);
plot_f2=plot(vec(x_f2), y_f2, seriestype=:scatter, markersize = 1, legend=false)
plot!(plot_f2, vec(x_f2_sort), μ_f2_hat, ribbon=vec(exp.(logσ_f2)), fillalpha=.5, linecolor=:red,
    fillcolor=:red, thickness_scaling=2, legend=false)
savefig("Q2_5_3(4)_f2")

θ_learned_f3 =train_nn_w_var_reg(target_f3, θ_init; lr = 1e-5, iters=100000);
x_f3,y_f3 = sample_batch(target_f3, 1000);
x_f3_sort=sort(x_f3, dims=2);
μ_f3_hat, logσ_f3=neural_net_w_var(x_f3_sort,θ_learned_f3);
plot_f3=plot(vec(x_f3), y_f3, seriestype=:scatter, markersize = 1, legend=false)
plot!(plot_f3, vec(x_f3_sort), μ_f3_hat, ribbon=vec(exp.(logσ_f3)), fillalpha=.5, linecolor=:red,
    fillcolor=:red, thickness_scaling=2, legend=false)
savefig("Q2_5_3(4)_f3")
