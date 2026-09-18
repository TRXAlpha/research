# Metacontrol afectiv endogen

[English version](README.md)

Cod și protocol de cercetare pentru a testa dacă un substrat afectiv autonom poate reorganiza cauzal cogniția unui agent bazat pe un model lingvistic.

Proiectul **nu** susține că sistemul artificial simte, este conștient sau posedă neurotransmițători biologici. Testează o ipoteză computațională mai precisă:

> În condiții de resurse limitate, o dinamică afectivă endogenă și constrânsă biologic poate produce regimuri cognitive reproductibile, atât adaptive, cât și maladaptive, care nu sunt explicate doar prin prompting, etichete emoționale externe sau un controler recurent generic cu aceeași capacitate.

## Diferența față de un chatbot care imită emoții

- Afectul este produs automat din evaluarea evenimentelor, eroarea de predicție, scopuri și variabile homeostatice sintetice.
- Agentul nu poate alege o emoție ca acțiune strategică.
- Starea persistă în afara contextului textual și evoluează pe o scară rapidă, respectiv una lentă.
- Afectul trebuie să modifice atenția, memoria, planificarea, explorarea, verificarea și plasticitatea, nu doar tonul răspunsului.
- Arhitectura cere intervenții cauzale, „leziuni” artificiale și canale independente de scriere și citire neuronală.
- Atât efectele benefice, cât și cele dăunătoare sunt rezultate experimentale valide.

## Ce face codul în prezent

Fluxul implementat este:

```text
eveniment observabil
  -> evaluare automată a relevanței pentru scop
  -> modificarea homeostaziei sintetice
  -> stare afectivă rapidă + dispoziție lentă
  -> coordonate neuronale continue
  -> intervenție în activările interne ale LLM-ului
  -> răspuns și rezultat
  -> evenimentul următor
```

Componentele principale sunt:

- `models.py` definește stările: evaluare, homeostazie, afect, modulație cognitivă și coordonate neuronale.
- `events.py` definește evenimente precum succes, eșec controlabil, eșec necontrolabil, amenințare, conflict, descoperire și odihnă. Evenimentele nu sunt etichete emoționale.
- `affect.py` aplică automat ecuațiile de tranziție. Agentul nu are o comandă de tipul „activează frica”.
- `modulation.py` transformă starea afectivă în predicții despre atenție, memorie, explorare, verificare, creativitate și eficiență. Include efecte neliniare, de exemplu curba în U inversat pentru activare.
- `neural.py` calibrează direcții neuronale de scriere și citire într-un LLM local înghețat și aplică intervenția printr-un hook într-un strat Transformer.
- `session.py` păstrează starea între interacțiuni, compară răspunsul de bază cu cel afectiv și salvează istoricul.
- `mvp.py` oferă interfața interactivă.
- `neural_benchmark.py` rulează un test în buclă închisă: rezultatul unei probleme generează automat evenimentul care afectează problema următoare.
- `agents.py`, `benchmark.py` și `experiment.py` implementează microlume procedurale și controlerele stateless, generic și afectiv.

LLM-ul folosit în MVP este SmolLM2-1.7B-Instruct, cu ponderile înghețate. Starea afectivă nu este adăugată în prompt. Sistemul scrie o combinație calibrată de direcții asemănătoare valenței, activării și dominanței într-un strat intern. Citirea este realizată într-un alt strat, cu alte exemple de calibrare, pentru a reduce circularitatea măsurării.

## Ce demonstrează și ce nu demonstrează MVP-ul

MVP-ul demonstrează că:

- o stare internă persistentă poate fi produsă automat din experiențe;
- această stare poate modifica în mod cauzal activările și unele răspunsuri ale unui LLM;
- aceeași intervenție poate avea efecte adaptive sau maladaptive;
- comparația poate fi făcută cu același prompt, seed și model de bază.

Nu demonstrează încă:

- existența unei experiențe subiective sau a conștiinței;
- că variabilele implementate sunt echivalente cu neurotransmițătorii reali;
- o îmbunătățire generală a raționamentului ori creativității;
- că sistemul actual reproduce fidel circuitele dopaminei sau ale amigdalei;
- că LoRA sau dinamica afectivă au fost deja antrenate.

## Structura repository-ului

- `docs/00-project-charter.md` — întrebarea științifică, domeniul și criteriile de succes.
- `docs/01-architecture.md` — substratul afectiv propus și legătura cu LLM-ul.
- `docs/02-experimental-protocol.md` — ipoteze, controale, metrici și statistică.
- `docs/03-prior-art.md` — limita actuală a noutății și cele mai apropiate lucrări.
- `docs/04-claims-ethics.md` — afirmații permise, siguranță și controlul antropomorfizării.
- `docs/05-brief-profesor-ro.md` — material în română pentru coordonatorul de neuroștiințe.
- `docs/06-mvp-guide.md` — instalare și testare interactivă.
- `docs/07-mvp-results.md` — rezultatele verificate și limitele MVP-ului.
- `docs/08-alarm-mvp.md` — intervenția minimală de alarmă și comparația prin „leziune” artificială.
- `ROADMAP.md` — planul de implementare în etape.
- `literature/prior-art.csv` — tabel structurat al literaturii.
- `src/affective_metacontrol/` — implementarea executabilă.
- `tests/` — teste de invariante și reproductibilitate.
- `references/references.bib` — bibliografia inițială.

## Rulare locală

Pe calculatorul de dezvoltare, mediul și modelul sunt deja instalate.

Pentru interfața interactivă, deschide `RUN_MVP.bat` sau rulează:

```powershell
cd C:\Users\chris\Documents\research
.\scripts\run_mvp.ps1
```

Demonstrație deterministă:

```powershell
.\scripts\run_mvp.ps1 --demo --max-new-tokens 64
```

Benchmark neuronal:

```powershell
.\scripts\run_neural_benchmark.ps1
```

Demonstrația emoției minimale de alarmă/amenințare:

```powershell
.\.venv\Scripts\python.exe -m affective_metacontrol.mvp --alarm-demo --gain 0.85 --alarm-gain 8.0
```

Pe Windows poate fi deschis direct `RUN_ALARM_DEMO.bat`. Demonstrația aplică două evenimente de amenințare, măsoară preferința modelului pentru acțiuni prudente și compară starea completă cu o „leziune” artificială în care numai canalul neuronal de alarmă este eliminat.

Teste automate:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m unittest discover -s tests -v
```

Comenzi în interfața MVP:

```text
/events
/event uncontrolled_failure
/event threat
/event success
/event rest
/state
/reset
/quit
```

## Cum trebuie generat datasetul de experiențe

Unitatea corectă nu este o pereche izolată întrebare-răspuns, ci o **traiectorie**. O traiectorie conține mai mulți pași legați cauzal, deoarece starea internă de la pasul curent depinde de experiențele anterioare.

Un exemplu de înregistrare JSONL:

```json
{
  "episode_id": "world_0042",
  "step": 3,
  "world_family": "causal_discovery",
  "observation": {"x": 5, "y": 1, "outcome": true},
  "goal": "identifică regula ascunsă folosind cel mult șase experimente",
  "event": "unexpected_success",
  "appraisal_inputs": {
    "novelty": 0.9,
    "goal_congruence": 0.8,
    "controllability": 0.7,
    "certainty": 0.5,
    "agency": 0.7,
    "urgency": 0.3,
    "prediction_error": 0.9
  },
  "state_before": {"valence": 0.0, "arousal": 0.2},
  "state_after": {"valence": 0.31, "arousal": 0.46},
  "available_actions": [[0, 0], [2, 5], [5, 1]],
  "chosen_action": [2, 5],
  "target_action": [2, 5],
  "task_score": 1.0,
  "cost": 0.1
}
```

Valorile complete ale stării trebuie salvate, dar nu trebuie transformate în propoziții precum „agentul este fericit” în prompt. Datasetul descrie lumea, scopul, predicția, acțiunea și consecința; substratul afectiv calculează starea.

Procedura recomandată:

1. Se construiesc familii de medii procedurale cu răspuns verificabil: descoperire cauzală, memorie cu interferență, planificare sub buget, explorare, verificarea ipotezelor și căutare creativă constrânsă.
2. Pentru fiecare seed se generează o traiectorie completă, nu rânduri independente.
3. Un evaluator determinist calculează succesul, costul, informația câștigată și eroarea de predicție.
4. Din aceste rezultate se calculează automat appraisal-ul și tranziția afectivă.
5. Ținta de învățare este furnizată de un solver, de o politică optimă sau de un model-profesor verificat prin evaluator; nu de etichete emoționale scrise manual.
6. Se salvează și traiectorii contrafactuale: aceeași lume și aceeași observație, dar cu istorii interne diferite.
7. Împărțirea train/validation/test se face după seed și familie de lume, nu aleatoriu după rând, pentru a preveni contaminarea.
8. Se păstrează aceleași traiectorii pentru controalele stateless, recurent generic și afectiv.

Primele date ar trebui să fie complet sintetice și verificabile. Datele conversaționale „emoționale” de pe internet ar antrena mai ales stilul verbal și ar compromite ipoteza principală.

## Este LoRA alegerea potrivită?

**Da, ca prim mecanism eficient de adaptare; nu, ca întregul sistem afectiv.**

LoRA păstrează modelul de bază înghețat și antrenează actualizări de rang mic. Este potrivit pentru un prototip cu resurse reduse și permite compararea sau dezactivarea rapidă a adaptoarelor. Însă un singur LoRA static învață o singură modificare stabilă și nu reprezintă singur o stare afectivă care se schimbă de la un moment la altul.

Arhitectura recomandată pentru proiect este:

```text
substrat afectiv persistent, neantrenat ca text
  -> vector intern continuu
  -> porți care controlează LoRA/FiLM sau un amestec de adaptoare
  -> modificarea procesării LLM-ului
```

Ordinea experimentală recomandată:

1. Activation steering-ul actual rămâne baseline-ul cauzal fără antrenare.
2. Se antrenează un LoRA mic pentru sensibilitate la semnalul intern, păstrând modelul de bază înghețat.
3. Se compară cu un LoRA identic ca număr de parametri, controlat de o stare recurentă generică.
4. Se testează un amestec de adaptoare LoRA sau un modul FiLM ale cărui porți sunt determinate continuu de starea afectivă.
5. Abia după rezultate robuste se justifică full fine-tuning-ul sau plasticitatea online.

Pentru GPU-ul NVIDIA T1200 de 4 GB, un pilot LoRA/QLoRA cu un model de 360M este cea mai sigură etapă inițială. Modelul de 1.7B poate necesita cuantizare la 4 biți, secvențe scurte, batch 1, gradient accumulation și gradient checkpointing. Instalarea curentă PyTorch este CPU-only, deci suportul CUDA trebuie configurat înainte de antrenare.

## Statut științific

Afirmația de noutate este încă o ipoteză de lucru și trebuie actualizată printr-o analiză sistematică. Starea VAD persistentă, activation steering-ul, appraisal-ul afectiv și raționarea dependentă de emoții există deja separat și în unele combinații apropiate.

Contribuția urmărită este mai îngustă: un sistem afectiv endogen, persistent și constrâns biologic, cu buclă bidirecțională între experiență, afect și cogniție, măsurători independente și efecte cognitive atât adaptive, cât și maladaptive, comparat cu un controler generic de aceeași capacitate.
