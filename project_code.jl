using CSV, DataFrames, SubsetSelection, StatsBase

st_train = DataFrame(CSV.File("C:/Users/wagnedg1/Documents/EP/strikers_train.csv"))

k=10;
X = st_train[!, Not(1:2)];; Y = st_train[!, 2];
X = Matrix(X); 
Sparse_Regressor = subsetSelection(OLS(), Constraint(k), Y, X)

Sparse_Regressor.indices

show(stdout, "text/plain", Sparse_Regressor.w)
show(stdout, "text/plain", Sparse_Regressor.indices)

# Compute predictions
st_test = DataFrame(CSV.File("C:/Users/wagnedg1/Documents/EP/strikers_test.csv"))
st_test = st_test[!, Not(1:2)];
st_test_mat = Matrix(st_test)
Y_pred = st_test_mat[:,Sparse_Regressor.indices]*Sparse_Regressor.w

CSV.write("C:/Users/wagnedg1/Documents/EP/julia_predictions_strikers_k10.csv",  Tables.table(Y_pred), writeheader=false)

# Right Wing Backs

rwb_train = DataFrame(CSV.File("C:/Users/wagnedg1/Documents/EP/rwb_train.csv"))

k=63;
X = rwb_train[!, Not(1:2)];; Y = rwb_train[!, 2];
X = Matrix(X); 
Sparse_Regressor = subsetSelection(OLS(), Constraint(k), Y, X)

Sparse_Regressor.indices
show(stdout, "text/plain", Sparse_Regressor.indices)

# Compute predictions
rwb_test = DataFrame(CSV.File("C:/Users/wagnedg1/Documents/EP/rwb_test.csv"))
rwb_test = rwb_test[!, Not(1:2)];
rwb_test_mat = Matrix(rwb_test)
Y_pred = rwb_test_mat[:,Sparse_Regressor.indices]*Sparse_Regressor.w

CSV.write("C:/Users/wagnedg1/Documents/EP/julia_predictions_rwbs_k63.csv",  Tables.table(Y_pred), writeheader=false)


