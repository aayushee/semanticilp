# SemanticILP 
This code is forked from https://github.com/allenai/semanticilp \


### Initializing the annotators 
The system is dependant on the set of annotators provided in CogCompNLP. In order to run annotators, download version 

The system is tested with [v3.1.22 of CogCompNLP](https://github.com/CogComp/cogcomp-nlp/releases/tag/v3.1.22). 
Download the package and run the annotator servers, on two different ports `PORT_NUMBER1` and `PORT_NUMBER2`.  
```bash 
# running the main annotators 
./pipeline/scripts/runWebserver.sh  --port PORT_NUMBER1 

```

Then you have to set the ports in SemanticILP. Open [`Constants.scala`](src/main/scala/org/allenai/ari/solvers/textilp/utils/Constants.scala) and set the ports.   
PORT_NUMBER1 = 9000 (for annotator server)
SOLVER_PORT = 9003 (for SemanticILP server)

**Note:** The annotators require good amount of memory: 
- CogComp-NLP pipeline takes up to 25GB

### Missing Dependencies 
Unfortunately some of our dependencies are not available publicly. But there is a hacky way to get around this issue. 
We have put these dependencies [here](https://drive.google.com/file/d/1eAcBoZOJ3GyB1Y_zcge_dRvILY6rJPFC/view?usp=sharing), which you have to put them in our ivy cache folder. 
In a typical machine this is where there should be located at: `~/.ivy2/cache/`.

### Running SemanticILP 
*Note:* here are the memory requirements: 
- SemanticILP solver: minimum around 8GB 
- Annotation Server (CogComp): minimum around 17GB 
*Note:* If you see an error like this: 
```
Caused by: java.lang.UnsatisfiedLinkError: no jscip-0.1.linux.x86_64.gnu.opt.spx in java.library.path
```
this means that the solver does not recognize the ILP binary files (common to linux). In that case, add the path to 
 your binary files, to your `LD_LIBRARY_PATH` variable. 
```
export LD_LIBRARY_PATH=path_to_lib_folder/
```

#### Run the solver over a network 
To the run the system over the network, run the following script: 
```
 > sbt 
 > project viz 
 > run SOLVER_PORT
```

And access it in this URL: 
```bash
http://SOLVER_DOMAIN:SOLVER_PORT
```
where `SOLVER_DOMAIN` is the domain of on which you're running the solver, `SOLVER_PORT` is the port on which 
the solver is running.
To stop it, just do Ctrl+D.

Once the server is up and running, it can be accessed from python code via http request.
Make sure to note the Solver Domain and Solver Port.

By default, the code will give results for 3 features. \
To run it for 2/4 features, make the following change and restart server with sbt: \
Open [`TextILPSolver.scala`](src/main/scala/org/allenai/ari/solvers/textilp/solvers/TextILPSolver.scala)
Set flags in lines 91 and 92:\
For 2 features, set   interSentFlag = 0, interParaFlag = 0 \
For 3 features, set   interSentFlag = 0, interParaFlag = 1 [default] \
For 4 features, set   interSentFlag = 1, interParaFlag = 1