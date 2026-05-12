import random

def generate_testcase(filename, num_atoms, num_rules, max_body_size):
    with open(filename, 'w') as f:
        # Commenti descrittivi
        f.write("c Testcase generato automaticamente per propagatore pesato\n")
        f.write(f"c Parametri: atomi={num_atoms}, regole={num_rules}\n")
        
        # Riga di definizione del problema
        f.write(f"p {num_atoms} {num_rules}\n")
        
        # Assegnamento iniziale (M): assegniamo casualmente circa il 5% degli atomi
        num_init = max(1, num_atoms // 20)
        init_atoms = random.sample(range(1, num_atoms + 1), num_init)
        init_lits = [a if random.random() > 0.5 else -a for a in init_atoms]
        
        # Riga di assegnamento terminata da 0
        f.write("a " + " ".join(map(str, init_lits)) + " 0\n")
        
        # Generazione delle regole
        for _ in range(num_rules):
            # Head della regola
            head = random.randint(1, num_atoms)
            if random.random() > 0.5:
                head = -head
                
            # Dimensione del body (k)
            k = random.randint(1, max_body_size)
            body_atoms = random.sample(range(1, num_atoms + 1), k)
            body_lits = [a if random.random() > 0.5 else -a for a in body_atoms]
            weights = [random.randint(1, 20) for _ in range(k)]
            
            # Bound B tarato per evitare regole banalmente impossibili o sempre vere
            total_weight = sum(weights)
            B = random.randint(min(weights), total_weight + 2)
            
            # Formattazione coppie (atomo, peso)
            rule_parts = [f"{lit} {w}" for lit, w in zip(body_lits, weights)]
            f.write(f"r {head} {B} {k} " + " ".join(rule_parts) + "\n")

if __name__ == "__main__":
    generate_testcase("small_test.txt", num_atoms=50, num_rules=20, max_body_size=5)
    generate_testcase("large_test.txt", num_atoms=10000, num_rules=50000, max_body_size=10)
    print("Testcase generati con successo!")