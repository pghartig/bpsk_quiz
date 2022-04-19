# bpsk_quiz

This repository is a sandbox for the BPSK demodulation quiz.

Results can be found in the "results" directory.

### Notes
* I was a little uncertain about the timing assumption mentioned in the problem statement. I mentioned methods for timing recovering in the code but assumed timing was found. 
* There is definitely some tuning that could be done to improve convergence rates. The provided "time" for convergence was 1
  second. I interpretted this as meaning up to 1e6 symbols and must not perform any computationally heavy tasks that would exceed 1
second.
* I had never actually implmented a PLL before (it was fun!) but I imagine that with some more work, convergence could be hastened with
  higher order methods and that stability could be improved by incorporating more sample memory into iterations.
* Some of the image saving code is sort of ugly but I wanted to ensure readers should easily run.
### Bonus Questions
* Using pilots would provide ambiguity resolution.
* Using a differential bpsk would ease implementation as phase ambiguity does not impede detection.
