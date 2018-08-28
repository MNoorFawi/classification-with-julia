
using Knet, RDatasets, DataFrames, Gadfly

# read data
data = readtable("bank-additional.csv", separator = ';');
describe(data)

########## Knet Logistic Regression ##############
atype = Array{Float32}; # atype = KnetArray{Float32} for gpu usage, Array{Float32} for cpu. 

# there are lots of String and Number columns that need to be encoded 
# because in Julia it's better to deal with numbers, we will encode String columns
# to new binary variables of 0/1 and we will scale the Number columns

# first change outcome variable y to 0 and 1, there are two methods
# subscribe(x) = 1.0(x .== "yes") # one for yes
data[:, :y] = map(Float64, data[:, :y] .== "yes");

x = DataFrame();
dataType = describe(data);
# separate String and Number values 
str = dataType[dataType[:eltype] .== String, :variable];
num = dataType[(dataType[:eltype] .== Float64) .| (dataType[:eltype] .== Int64), :variable];

# encode each column's String value to a new binary column of 0 or 1
# we here use "_" because there are repetitive values and we need to know which is which
for i in str
	dict = unique(data[:, i])
	for key in dict
		x[:, [Symbol(key .* "_" .* String(i))]] = 1.0(data[:, i] .== key)
	end
end

# we scale the Number columns except for the y variable 
num2 = setdiff(num, [:y])
d = DataFrame();
for i in num2
	d[:, i] = (data[:, i]- minimum(data[:, i])) / (maximum(data[:, i]) - minimum(data[:, i]))
end

# we can also define a function that changes all columns at once 

# function applycol!(fun::Function, df::DataFrame)
#	 for(name, col) in eachcol(df)
#	 	 df[name] = fun(col)
# 	 end
# end
# applycol!(col -> col ./ sum(col), df)

# then we combine everything together 
x = hcat(x, d) 
x[:y] = map(Float64, data[:, :y] .== "yes");

size(x)

# then we define our model;
# the model equation
predict(w, x) = w[1] * x .+ w[2];
# the model loss function 
function loss(w, x, y)
    yhat = sigm.(predict(w, x))
    return -sum(y .* log.(yhat) + (1-y) .* log.(1-yhat))
end;

lossgradient  = grad(loss);

# the model train function
function train(w, btrain; lr=1e-6, epochs=5)
    tloss = []
    for epoch = 1:epochs
        eloss = 0
        for (x,y) in btrain
            eloss += loss(w, x, y)
            g = lossgradient(w, x, y)
            for i = 1:length(w)
                w[i] -= lr * g[i]
            end
        end
        push!(tloss, eloss/length(btrain))
    end
    
    return w, tloss
end;

# a function to measure accuracy
Accuracy(w, x, yreal) = sum((sign.(predict(w, x)) + 1) / 2 .== yreal) / length(yreal);

# we then randomly splt the data to train and test data with 90:10 ratio
splits = round(Int, 0.1 * size(x, 1));
shuffled = randperm(size(x, 1));
xtrain, ytrain = map(atype, [Array(x[shuffled[splits + 1:end], 1:end-1])', Array(x[shuffled[splits + 1:end], end:end])']);
xtest, ytest = map(atype, [Array(x[shuffled[1:splits], 1:end-1])', Array(x[shuffled[1:splits], end:end])']);
# check that both data are of the same distribution
sum(ytrain) / length(ytrain), sum(ytest) / length(ytest)
# size of each one
size(xtrain), size(xtest)
# special Knet iterable that treat data in batchesl useful with very big data 
btrain = minibatch(xtrain, ytrain, 50; shuffle = true);
# initialize coefficients
w = map(atype, Any[randn(1, size(xtrain, 1)), zeros(Float32, 1, 1)]);
# accuracy before training; random accuracy 
Accuracy(w, xtest, ytest)

# train the model 
w, Loss = train(w, btrain; epochs = 30, lr = 1e-2);
# plot how the Loss has been decreasing 
d = DataFrame(indx = [i for i in 1:length(Loss)], loss = Loss)
plot(d, x = :indx, y = :loss, Geom.point, Scale.y_continuous(minvalue = minimum(Loss), maxvalue = maximum(Loss)))
# accuracy in train and test data
Accuracy(w, xtrain, ytrain)
Accuracy(w, xtest, ytest)

############### Decision Tree ####################
using DecisionTree
# prepare data 
trfeatures = float.(xtrain');
trlabels   = string.(ytrain');
tsfeatures = float.(xtest');
tslabels   = string.(ytest');
# define the model and fit it 
model = DecisionTreeClassifier(max_depth = 2)
fit!(model, trfeatures, trlabels[:, 1]) 
println(get_classes(model))
# predict on test data
TreePred = [DecisionTree.predict(model, tsfeatures[i, :]) for i in 1:size(tsfeatures, 1)];
# measure accuracy on test data
sum(TreePred .== tslabels) / length(tslabels)
# get the probability of each label
predProb = [DecisionTree.predict_proba(model, trfeatures[i, :])[2] for i in 1:size(trfeatures, 1)];

# Random Forest using 7 random features, 10 trees, 0.9 portion of samples per tree, and a maximum tree depth of 6
model = DecisionTree.build_forest(trlabels[:, 1], trfeatures, 2, 10, 0.5, 6)
# predict labels and examine accuracy on test data
forestPred = [apply_forest(model, tsfeatures[i, :]) for i in 1:size(tsfeatures, 1)]
sum(forestPred .== tslabels) / length(tslabels)
# get the probability of each label
[apply_forest_proba(model, tsfeatures[i, :], ["0.0", "1.0"]) for i in 1:size(tsfeatures, 1)]

