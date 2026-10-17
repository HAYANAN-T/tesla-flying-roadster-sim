# Tesla "flying" Roadster in traffic: physics model and traffic simulation

Code behind the article "Tesla's Flying Roadster in Real Traffic: What My AI Simulation Found".

## What is inside

- `src/physics.py` – cold-gas thruster model: an air tank emptying through choked nozzles. Calibrated to the figures reported at Tesla's Oct 15, 2026 reveal (about 600 L of tanks, about 250 kg of air, more than 80 kN of thrust).
- `src/traffic_sim.py` – a 1 km single-lane ring road. Cars follow the Intelligent Driver Model. One car can hop over the traffic ahead. Three ways of planning the hop are compared: no AI, AI prediction, and AI prediction with car-to-car communication (V2V).
- `src/run_experiments.py` – runs 300 hop attempts for each traffic density (15 to 45 cars per km) and each planner, and writes `results/`.
- `visualization/plots.py` – draws every simulation figure in `images/`.

## How to run

    pip install -r requirements.txt
    python src/run_experiments.py --rerun     # about 10-20 minutes on a laptop
    python visualization/plots.py

## Main assumptions (change them in the code)

- Hop: 1.5 seconds in the air, can reach a gap up to 80 m ahead.
- Landing error: 1.0 m without AI, 0.3 m with AI.
- Each nearby human driver has a 35% chance of braking hard (3 m/s² for 1.5 s) when a car jumps near them. Connected (V2V) cars are warned in advance and do not startle.
- The AI only hops if it predicts at least 1 second of gap in front of and behind the landing spot.
- Vehicle mass is unknown, so the physics is run for 1,500, 1,800 and 2,100 kg.

These are modelling choices, not measured facts about the real car. Results are a thought experiment with real physics and a standard traffic model.

## License

MIT
