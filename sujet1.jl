using JuMP, GLPK
using Random

# --- Données ---
m = 10
n = 10
p_rares, q_communes = 3, 3
#alphaR, alphaC = 0.5, 0.9

raw_proba = [
    1 4 3 0.4; 1 5 3 0.3; 1 6 2 0.4; 1 8 6 0.3; 1 8 7 0.2; 1 9 5 0.2; 1 9 6 0.4;
    2 2 7 0.2; 2 3 7 0.4; 2 4 7 0.2; 2 5 9 0.4; 2 5 10 0.3; 2 7 2 0.5; 2 9 6 0.2; 2 9 7 0.2;
    3 2 4 0.2; 3 2 5 0.3; 3 2 7 0.4; 3 3 7 0.4; 3 4 7 0.5; 3 5 9 0.2; 3 5 10 0.4; 3 9 2 0.3;
    4 1 2 0.3; 4 1 4 0.3; 4 3 2 0.4; 4 4 3 0.4; 4 5 1 0.3; 4 5 3 0.2; 4 5 5 0.2; 4 6 2 0.2; 
    4 6 3 0.4; 4 7 5 0.4; 4 7 9 0.3; 4 8 7 0.2; 4 8 9 0.5; 4 9 7 0.4;
    5 1 10 0.4; 5 2 1 0.3; 5 2 10 0.3; 5 3 5 0.5; 5 6 3 0.4; 5 6 6 0.2; 5 6 7 0.2; 
    5 7 5 0.4; 5 7 9 0.5; 5 8 9 0.4; 5 9 2 0.5; 5 9 3 0.2; 5 9 4 0.4; 5 9 5 0.4;
    6 1 3 0.4; 6 1 4 0.4; 6 1 6 0.5; 6 1 7 0.3; 6 1 8 0.3; 6 2 9 0.2; 6 3 5 0.4; 
    6 3 9 0.4; 6 3 10 0.2; 6 4 9 0.3; 6 5 5 0.4; 6 8 1 0.5; 6 9 1 0.2; 6 10 3 0.2
]

# Initialisation du tenseur 3D avec des 0 (le comportement "default 0" de AMPL)
proba = zeros(Float64, p_rares + q_communes, m, n)

# Remplissage du tenseur proba[k, i, j]
for row in 1:size(raw_proba, 1)
    k = Int(raw_proba[row, 1])
    i = Int(raw_proba[row, 2])
    j = Int(raw_proba[row, 3])
    val = raw_proba[row, 4]
    
    proba[k, i, j] = val
end


ER = 1:p_rares
EC = (p_rares+1):(p_rares+q_communes)

# Initialisation des coÃ»ts selon la logique AMPL
c_cost = zeros(Int, m, n)
for i in 1:m, j in 1:n
    if i<=3 && j<=3;     c_cost[i,j] = 6
    elseif i<=3 && j<=7; c_cost[i,j] = 4
    elseif i<=3 && j<=10; c_cost[i,j] = 8
    elseif i<=7 && j<=3; c_cost[i,j] = 5
    elseif i<=7 && j<=7; c_cost[i,j] = 3
    elseif i<=7 && j<=10; c_cost[i,j] = 7
    elseif i<=10 && j<=3; c_cost[i,j] = 4
    elseif i<=10 && j<=7; c_cost[i,j] = 6
    else;                 c_cost[i,j] = 5
    end
end

#fonctions
#question 3

function resoudre(alphaR, alphaC; C = c_cost, P = proba, verbose = true)
    m, n = size(C)
    model = Model(GLPK.Optimizer)
    verbose && set_attribute(model, "msg_lev", GLPK.GLP_MSG_ALL)
    set_time_limit_sec(model, 120)
    
    # Variables
    @variable(model, x[1:m, 1:n], Bin)   # x[i,j] = 1 si la parcelle est centrale
    @variable(model, y[1:m, 1:n], Bin)   # y[i,j] = 1 si la parcelle est tampon
    
    # Objectif : coût total des parcelles protégées
    @objective(model, Min, sum(C[i,j] * (x[i,j] + y[i,j]) for i in 1:m, j in 1:n))
    
    # C1 : une parcelle a au plus un rôle
    @constraint(model, [i in 1:m, j in 1:n], x[i,j] + y[i,j] <= 1)
    
    # C2 : une parcelle du bord n'a pas 8 voisines donc n'est pas centrale
    for i in 1:m, j in 1:n
        if (i == 1 || i == m || j == 1 || j == n )
            fix(x[i,j], 0; force = true)
        end
    end
    
    # C3 : si (i,j) est centrale, ses 8 voisines sont dans la réserve
    @constraint(model, [i in 2:m-1, j in 2:n-1, di in -1:1, dj in -1:1; (di, dj) != (0, 0)],
    x[i+di, j+dj] + y[i+di, j+dj] >= x[i,j])

    # C4 : sum log(1-p) * s <= log(1-alpha)
    @constraint(model, [k in ER],
    sum(log(1 - P[k,i,j]) * x[i,j] for i in 1:m, j in 1:n if P[k,i,j] > 0)
    <= log(1 - alphaR))
    @constraint(model, [k in EC],
    sum(log(1 - P[k,i,j]) * (x[i,j] + y[i,j]) for i in 1:m, j in 1:n if P[k,i,j] > 0)
    <= log(1 - alphaC))
    
    optimize!(model)
    return model, x, y
end

#question 5
function resoudre_v(B; C = c_cost, P = proba, verbose = false)
    m, n = size(C)
    model = Model(GLPK.Optimizer) #pour le moment j'ai pas lisence GUROBI
    verbose && set_attribute(model, "msg_lev", GLPK.GLP_MSG_ALL)
    set_time_limit_sec(model, 120)

    @variable(model, x[1:m, 1:n], Bin)   # x_ij = 1 si centrale
    @variable(model, y[1:m, 1:n], Bin)   # y_j = 1 si tampon

    # (même contrainte pour zone)
    @constraint(model, [i in 1:m, j in 1:n], x[i,j] + y[i,j] <= 1)
    for i in 1:m, j in 1:n
        if i == 1 || i == m || j == 1 || j == n
            fix(x[i,j], 0; force = true)
        end
    end
    @constraint(model, [i in 2:m-1, j in 2:n-1, di in -1:1, dj in -1:1; (di, dj) != (0, 0)],
        x[i+di, j+dj] + y[i+di, j+dj] >= x[i,j])

    # contrainte Budget 
    @constraint(model, sum(C[i,j] * (x[i,j] + y[i,j]) for i in 1:m, j in 1:n) <= B)

    # Décomposition du produit, espèce par espèce
    q_final = []
    for k in 1:size(P, 1)
        parcelles = [(i, j) for i in 1:m, j in 1:n if P[k,i,j] > 0]   # S_k
        q_prec = 1.0                                                  # q_{k,0} = 1
        for (i, j) in parcelles
            s = k in ER ? x[i,j] : x[i,j] + y[i,j]                  
            q = @variable(model, lower_bound = 0, upper_bound = 1)    # q_{k,t}
            w = @variable(model, lower_bound = 0, upper_bound = 1)    # w_{k,t} = q_{k,t-1} * s
            @constraint(model, w <= q_prec)               # si s = 1 : w <= q
            @constraint(model, w <= s)                    # si s = 0 : w = 0
            @constraint(model, w >= q_prec - (1 - s))     # si s = 1 : w >= q
            @constraint(model, q == q_prec - P[k,i,j] * w)
            q_prec = q
        end
        push!(q_final, q_prec)                            # q_{k,T_k} = P(k disparaît)
    end

    @objective(model, Max, sum(1 - qf for qf in q_final))
    optimize!(model)
    return model, x, y
end


function survie(k, xv, yv)
    s = 1.0
    for i in 1:m, j in 1:n
        protege = k in ER ? xv[i,j] > 0.5 : xv[i,j] + yv[i,j] > 0.5
        if proba[k,i,j] > 0 && protege
            s *= 1 - proba[k,i,j]
        end
    end
    return 1 - s
end


function generer_instance(m, n; densite = 0.15, graine = 1)
    rng = MersenneTwister(graine)            # graine fixée
    C = rand(rng, 1:10, m, n)                # coûts entiers entre 1 et 10
    P = zeros(6, m, n)
    for k in 1:6, i in 1:m, j in 1:n
        if rand(rng) < densite               # l'espèce k vit dans (i,j) ?
            P[k,i,j] = rand(rng, 1:5) / 10   # probabilité entre 0.1 et 0.5
        end
    end
    return C, P
end

function etude_taille()
    for taille in [10, 15, 20, 25, 30]
        for graine in 1:3
            C, P = generer_instance(taille, taille; graine = graine)
            model, x, y = resoudre(0.8, 0.8; C = C, P = P, verbose = false)
            cout = has_values(model) ? objective_value(model) : NaN
            println("taille = $taille, graine = $graine : ", termination_status(model),
                    ", coût = $cout, temps = ", round(solve_time(model), digits = 2), " s")
        end
    end
end

function test_v()
    for B in [119, 211, 327]
        model, x, y = resoudre_v(B)
        xv, yv = value.(x), value.(y)
        esperance = sum(survie(k, xv, yv) for k in 1:6)
        println("B = $B : ", termination_status(model),
                ", objectif = ", round(objective_value(model), digits = 4),
                ", espérance recalculée = ", round(esperance, digits = 4))
    end
end

function main()
    cas = [(0.5, 0.5), (0.9, 0.5), (0.5, 0.9), (0.8, 0.6)]
    for (aR, aC) in cas
        println("\n================ alphaR = $aR, alphaC = $aC ================")
        model, x, y = resoudre(aR, aC)
        xv, yv = value.(x), value.(y)

        println("Statut = ", termination_status(model))
        println("Coût   = ", objective_value(model))
        println("Temps  = ", round(solve_time(model), digits = 3), " s")
        println("Survie = ", [round(survie(k, xv, yv), digits = 2) for k in 1:6])

        for i in 1:m     
            println(join([xv[i,j] > 0.5 ? "#" : yv[i,j] > 0.5 ? "+" : "." for j in 1:n]))
        end
    end
end 

main()
etude_taille()