def genera_regole_sudoku_nxn(K):
    """
    Genera le regole per un Sudoku N x N, dove N = K * K.
    Esempio: K=2 -> Sudoku 4x4. K=3 -> Sudoku 9x9. K=4 -> Sudoku 16x16.
    """
    N = K * K
    regole = []
    
    # Atomo speciale usato come head. Usiamo un numero altissimo per non 
    # sovrapporsi con le variabili generate dalla scacchiera.
    H_TRUE = N * N * N + 1
    H_FALSE = N * N * N + 2

    def get_var(r, c, v):
        """
        Mappatura univoca sicura per qualsiasi dimensione N.
        Calcola l'indice lineare in 3 dimensioni e aggiunge 1 per non avere la variabile 0.
        """
        return (r - 1) * (N * N) + (c - 1) * N + (v - 1) + 1

    def add_at_least_one(vars):
        """Almeno una delle variabili deve essere vera."""
        k = len(vars)
        corpo = []
        for var in vars:
            corpo.extend([str(var), "1"])
        
        regola = f"{H_TRUE} 1 {k} " + " ".join(corpo)
        regole.append(regola)
    
    def add_at_most_one(vars):
        """Almeno una delle variabili deve essere vera."""
        k = len(vars)
        corpo = []
        for var in vars:
            corpo.extend([str(var), "1"])
        
        regola = f"{H_FALSE} 2 {k} " + " ".join(corpo)
        regole.append(regola)



    def add_exactly_one(vars):
        """Combina le due precedenti."""
        add_at_least_one(vars)
        add_at_most_one(vars)

    # 1. Vincolo di Cella: Ogni cella (r,c) deve avere esattamente un valore
    for r in range(1, N + 1):
        for c in range(1, N + 1):
            vars = [get_var(r, c, v) for v in range(1, N + 1)]
            add_exactly_one(vars)

    # 2. Vincolo di Riga: Ogni riga r deve avere il valore v esattamente una volta
    for r in range(1, N + 1):
        for v in range(1, N + 1):
            vars = [get_var(r, c, v) for c in range(1, N + 1)]
            add_exactly_one(vars)

    # 3. Vincolo di Colonna: Ogni colonna c deve avere il valore v esattamente una volta
    for c in range(1, N + 1):
        for v in range(1, N + 1):
            vars = [get_var(r, c, v) for r in range(1, N + 1)]
            add_exactly_one(vars)

    # 4. Vincolo di Riquadro (Box KxK): Ogni box deve avere il valore v esattamente una volta
    for br in range(K):
        for bc in range(K):
            for v in range(1, N + 1):
                vars = []
                for i in range(1, K + 1):
                    for j in range(1, K + 1):
                        r = br * K + i
                        c = bc * K + j
                        vars.append(get_var(r, c, v))
                add_exactly_one(vars)

    return regole

if __name__ == "__main__":
    # Sostituisci il parametro per cambiare dimensione. 
    # 2 = 4x4, 3 = 9x9, 4 = 16x16
    K = 3
    N=K*K
    regole = genera_regole_sudoku_nxn(K)

    
    # Per non stampare migliaia di righe a schermo, potresti scriverle su file:
    with open(f"sudoku_{K*K}x{K*K}.in", "w") as f:
        f.write(f"p {N*N*N+2} {len(regole)}\n")
        for r in regole:
            f.write("r " + r + "\n")
        f.write(f"a {N*N*N+1} {-(N*N*N+2) }")
    print(f"Generazione completata: {len(regole)} regole generate per un Sudoku {K*K}x{K*K}.")