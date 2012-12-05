using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using System.Drawing;

namespace TSP
{

    class ProblemAndSolver
    {

        private class TSPSolution
        {
            /// <summary>
            /// we use the representation [cityB,cityA,cityC] 
            /// to mean that cityB is the first city in the solution, cityA is the second, cityC is the third 
            /// and the edge from cityC to cityB is the final edge in the path.  
            /// You are, of course, free to use a different representation if it would be more convenient or efficient 
            /// for your node data structure and search algorithm. 
            /// </summary>
            public ArrayList
                Route;

            public TSPSolution(ArrayList iroute)
            {
                Route = new ArrayList(iroute);
            }


            /// <summary>
            /// Compute the cost of the current route.  
            /// Note: This does not check that the route is complete.
            /// It assumes that the route passes from the last city back to the first city. 
            /// </summary>
            /// <returns></returns>
            public double costOfRoute()
            {
                // go through each edge in the route and add up the cost. 
                int x;
                City here;
                double cost = 0D;

                for (x = 0; x < Route.Count - 1; x++)
                {
                    here = Route[x] as City;
                    cost += here.costToGetTo(Route[x + 1] as City);
                }

                // go from the last city to the first. 
                here = Route[Route.Count - 1] as City;
                cost += here.costToGetTo(Route[0] as City);
                return cost;
            }
        }

        #region Private members 

        /// <summary>
        /// Default number of cities (unused -- to set defaults, change the values in the GUI form)
        /// </summary>
        // (This is no longer used -- to set default values, edit the form directly.  Open Form1.cs,
        // click on the Problem Size text box, go to the Properties window (lower right corner), 
        // and change the "Text" value.)
        private const int DEFAULT_SIZE = 25;

        private const int CITY_ICON_SIZE = 5;

        // For normal and hard modes:
        // hard mode only
        private const double FRACTION_OF_PATHS_TO_REMOVE = 0.20;

        /// <summary>
        /// the cities in the current problem.
        /// </summary>
        private City[] Cities;
        /// <summary>
        /// a route through the current problem, useful as a temporary variable. 
        /// </summary>
        private ArrayList Route;
        /// <summary>
        /// best solution so far. 
        /// </summary>
        private TSPSolution bssf; 

        /// <summary>
        /// how to color various things. 
        /// </summary>
        private Brush cityBrushStartStyle;
        private Brush cityBrushStyle;
        private Pen routePenStyle;


        /// <summary>
        /// keep track of the seed value so that the same sequence of problems can be 
        /// regenerated next time the generator is run. 
        /// </summary>
        private int _seed;
        /// <summary>
        /// number of cities to include in a problem. 
        /// </summary>
        private int _size;

        /// <summary>
        /// Difficulty level
        /// </summary>
        private HardMode.Modes _mode;

        /// <summary>
        /// random number generator. 
        /// </summary>
        private Random rnd;
        #endregion

        #region Public members
        public int Size
        {
            get { return _size; }
        }

        public int Seed
        {
            get { return _seed; }
        }
        #endregion

        #region Constructors
        public ProblemAndSolver()
        {
            this._seed = 1; 
            rnd = new Random(1);
            this._size = DEFAULT_SIZE;

            this.resetData();
        }

        public ProblemAndSolver(int seed)
        {
            this._seed = seed;
            rnd = new Random(seed);
            this._size = DEFAULT_SIZE;

            this.resetData();
        }

        public ProblemAndSolver(int seed, int size)
        {
            this._seed = seed;
            this._size = size;
            rnd = new Random(seed); 
            this.resetData();
        }
        #endregion

        #region Private Methods

        /// <summary>
        /// Reset the problem instance.
        /// </summary>
        private void resetData()
        {

            Cities = new City[_size];
            Route = new ArrayList(_size);
            bssf = null;

            if (_mode == HardMode.Modes.Easy)
            {
                for (int i = 0; i < _size; i++)
                    Cities[i] = new City(rnd.NextDouble(), rnd.NextDouble());
            }
            else // Medium and hard
            {
                for (int i = 0; i < _size; i++)
                    Cities[i] = new City(rnd.NextDouble(), rnd.NextDouble(), rnd.NextDouble() * City.MAX_ELEVATION);
            }

            HardMode mm = new HardMode(this._mode, this.rnd, Cities);
            if (_mode == HardMode.Modes.Hard)
            {
                int edgesToRemove = (int)(_size * FRACTION_OF_PATHS_TO_REMOVE);
                mm.removePaths(edgesToRemove);
            }
            City.setModeManager(mm);

            cityBrushStyle = new SolidBrush(Color.Black);
            cityBrushStartStyle = new SolidBrush(Color.Red);
            routePenStyle = new Pen(Color.Blue,1);
            routePenStyle.DashStyle = System.Drawing.Drawing2D.DashStyle.Solid;
        }

        #endregion

        #region Public Methods

        /// <summary>
        /// make a new problem with the given size.
        /// </summary>
        /// <param name="size">number of cities</param>
        //public void GenerateProblem(int size) // unused
        //{
        //   this.GenerateProblem(size, Modes.Normal);
        //}

        /// <summary>
        /// make a new problem with the given size.
        /// </summary>
        /// <param name="size">number of cities</param>
        public void GenerateProblem(int size, HardMode.Modes mode)
        {
            this._size = size;
            this._mode = mode;
            resetData();
        }

        /// <summary>
        /// return a copy of the cities in this problem. 
        /// </summary>
        /// <returns>array of cities</returns>
        public City[] GetCities()
        {
            City[] retCities = new City[Cities.Length];
            Array.Copy(Cities, retCities, Cities.Length);
            return retCities;
        }

        /// <summary>
        /// draw the cities in the problem.  if the bssf member is defined, then
        /// draw that too. 
        /// </summary>
        /// <param name="g">where to draw the stuff</param>
        public void Draw(Graphics g)
        {
            float width  = g.VisibleClipBounds.Width-45F;
            float height = g.VisibleClipBounds.Height-45F;
            Font labelFont = new Font("Arial", 10);

            // Draw lines
            if (bssf != null)
            {
                // make a list of points. 
                Point[] ps = new Point[bssf.Route.Count];
                int index = 0;
                foreach (City c in bssf.Route)
                {
                    if (index < bssf.Route.Count -1)
                        g.DrawString(" " + index +"("+c.costToGetTo(bssf.Route[index+1]as City)+")", labelFont, cityBrushStartStyle, new PointF((float)c.X * width + 3F, (float)c.Y * height));
                    else 
                        g.DrawString(" " + index +"("+c.costToGetTo(bssf.Route[0]as City)+")", labelFont, cityBrushStartStyle, new PointF((float)c.X * width + 3F, (float)c.Y * height));
                    ps[index++] = new Point((int)(c.X * width) + CITY_ICON_SIZE / 2, (int)(c.Y * height) + CITY_ICON_SIZE / 2);
                }

                if (ps.Length > 0)
                {
                    g.DrawLines(routePenStyle, ps);
                    g.FillEllipse(cityBrushStartStyle, (float)Cities[0].X * width - 1, (float)Cities[0].Y * height - 1, CITY_ICON_SIZE + 2, CITY_ICON_SIZE + 2);
                }

                // draw the last line. 
                g.DrawLine(routePenStyle, ps[0], ps[ps.Length - 1]);
            }

            // Draw city dots
            foreach (City c in Cities)
            {
                g.FillEllipse(cityBrushStyle, (float)c.X * width, (float)c.Y * height, CITY_ICON_SIZE, CITY_ICON_SIZE);
            }

        }

        /// <summary>
        ///  return the cost of the best solution so far. 
        /// </summary>
        /// <returns></returns>
        public double costOfBssf ()
        {
            if (bssf != null)
                return (bssf.costOfRoute());
            else
                return -1D; 
        }

        /// <summary>
        ///  solve the problem.  This is the entry point for the solver when the run button is clicked
        /// right now it just picks a simple solution. 
        /// </summary>
        public void solveProblem()
        {
            // start our timer
            System.Diagnostics.Stopwatch timer = new System.Diagnostics.Stopwatch();
            timer.Start();

            int x;
            Route = new ArrayList(); 
            // this is the trivial solution. 
            for (x = 0; x < Cities.Length; x++)
            {
                Route.Add( Cities[Cities.Length - x -1]);
            }
            // call this the best solution so far.  bssf is the route that will be drawn by the Draw method. 
            bssf = new TSPSolution(Route);

            Program.MainForm.tbCostOfTour.Text = " " + bssf.costOfRoute();
            Program.MainForm.tbElapsedTime.Text = Convert.ToString(timer.Elapsed);

            // do a refresh. 
            Program.MainForm.Invalidate();
        }   
        #endregion


        private class state
        {
            public double[,] costMatrix;
            public int[] inEdges;
            public int[] outEdges;
            public double cost;
            public int numEdges;
            public int numAddedEdges;

            public state(int edges)
            {
                cost = 0;
                numEdges = edges;
                inEdges = new int[edges];
                outEdges = new int [edges];
                for (int i=0; i < numEdges; i++)
                {
                    inEdges[i] = -1;
                    outEdges[i] = -1;
                }
                costMatrix = new double[edges, edges];
                numAddedEdges = 0;
            }
            public state(state other)
            {
                cost = other.cost;
                numEdges = other.numEdges;
                inEdges = new int[numEdges];
                outEdges = new int[numEdges];
                costMatrix = new double[numEdges, numEdges];
                for (int i = 0; i < numEdges; i++)
                {
                    inEdges[i] = other.inEdges[i];
                    outEdges[i] = other.outEdges[i];
                    for (int j = 0; j < numEdges; j++)
                    {
                        costMatrix[i, j] = other.costMatrix[i, j];
                    }
                }
                numAddedEdges = other.numAddedEdges;
            }

        }

        private void reduce(state cost)
        {
            // for each row
            for (int i = 0; i < cost.numEdges; i++)
            {
                // skip row if all infinities because edge already included
                if (cost.outEdges[i] != -1) continue;

                // find the min
                double min = double.PositiveInfinity;
                for (int j = 0; j < cost.numEdges; j++)
                {
                    if (cost.costMatrix[i,j] < min) min = cost.costMatrix[i,j];
                }

                // if the min is not 0 (already reduced)
                if (min != 0)
                {
                    // add to cost
                    cost.cost += min;
                    // subtract from all other entries
                    for (int j = 0; j < cost.numEdges; j++)
                    {
                        cost.costMatrix[i,j] -= min;
                    }
                }
            }

            // Repeat for each column
            for (int i = 0; i < cost.numEdges; i++)
            {
                // skip row if all infinities because edge already included
                if (cost.inEdges[i] != -1) continue;

                // find the min
                double min = double.PositiveInfinity;
                for (int j = 0; j < cost.numEdges; j++)
                {
                    if (cost.costMatrix[j,i] < min) min = cost.costMatrix[j,i];
                }

                // if the min is not 0 (already reduced)
                if (min != 0)
                {
                    // add to cost
                    cost.cost += min;
                    // subtract from all other entries
                    for (int j = 0; j < cost.numEdges; j++)
                    {
                        cost.costMatrix[j,i] -= min;
                    }
                }
            }
        }

        private double worstCase(state cost, int city1, int city2)
        {
            // will be min in row + min in column
            double ret = 0;
            double min = double.PositiveInfinity;
            for (int i=0; i < cost.numEdges; i++){
                if (i != city2 && cost.costMatrix[city1,i] < min) 
                    min = cost.costMatrix[city1,i];
            }
            ret += min;
            min = double.PositiveInfinity;
            for (int j=0; j < cost.numEdges; j++){
                if (j != city1 && cost.costMatrix[j,city2] < min)
                    min = cost.costMatrix[j,city2];
            }
            ret += min;
            ret += cost.cost;
            return ret;
        }

        private state includeEdge(state cost, int city1, int city2)
        {
            state ret = new state(cost);
            ret.cost += cost.costMatrix[city1,city2];

            // infinite out row
            for(int i=0; i < ret.numEdges; i++)
            {
                ret.costMatrix[i,city2] = double.PositiveInfinity;
            }

            // infinite out column
            for (int j=0; j < ret.numEdges; j++)
            {
                ret.costMatrix[city1,j] = double.PositiveInfinity;
            }

            // update in edges and out edges
            ret.outEdges[city1] = city2;
            ret.inEdges[city2] = city1;
            ret.numAddedEdges ++;

            // prevent cycles
            if (ret.numAddedEdges < ret.numEdges - 1)
            {
                // first, find the city in this subset that doesn't have an outgoing edge yet
                int lastCity = city1;
                while (ret.outEdges[lastCity] != -1)
                {
                    lastCity = ret.outEdges[lastCity];
                }

                // now, take that city and make sure it won't try to go into anything
                // already in the cycle.
                int prevCity = ret.inEdges[city2];
                while (prevCity != -1)
                {
                    // walk up the line of cities we just added
                    ret.costMatrix[lastCity,prevCity] = double.PositiveInfinity;
                    prevCity = ret.inEdges[prevCity];
                }
            }
            
            // call reduce
            reduce(ret);
            return ret;
        }

        private void initBSSF()
        {
            int x;
            ArrayList init = new ArrayList();
            List<int> fix = new List<int>();
            // this is the trivial solution. 
            for (x = 0; x < Cities.Length; x++)
            {
                init.Add(Cities[Cities.Length - x - 1]);
                if (x != 0 && Cities[Cities.Length - x].costToGetTo(Cities[Cities.Length - x - 1]) == double.PositiveInfinity)
                    fix.Add(x); // fix city at location x.
            }
            Random m = new Random();
            for (int i = 0; i < fix.Count; i++)
            {
                City fixPrev = init[(fix[i]-1) % Cities.Length] as City;
                City fixMid = init[fix[i]] as City;
                City fixNext = init[(fix[i] + 1) % Cities.Length] as City;
                City swapPrev;
                City swapMid;
                City swapNext;
                int swapIndex;
                do
                {
                    swapIndex = m.Next(1,Cities.Length - 1);
                    swapPrev = init[(swapIndex - 1) % Cities.Length] as City;
                    swapMid = init[swapIndex % Cities.Length] as City;
                    swapNext = init[(swapIndex + 1) % Cities.Length] as City;
                } while (fixPrev.costToGetTo(swapMid) == double.PositiveInfinity ||
                swapMid.costToGetTo(fixNext) == double.PositiveInfinity ||
                swapPrev.costToGetTo(fixMid) == double.PositiveInfinity ||
                fixMid.costToGetTo(swapNext) == double.PositiveInfinity);

                // found an agreeable swap, fix.
                init[fix[i]] = swapMid;
                init[swapIndex] = fixNext;
            }

            // call this the best solution so far.  bssf is the route that will be drawn by the Draw method. 
            bssf = new TSPSolution(init);
            

            // try 2 other random solutions
            for (int j = 0; j < 2; j++)
            {
                // randomize
                for (int i = 1; i < Cities.Length; i++)
                {
                    City fixPrev = init[(i - 1) % Cities.Length] as City;
                    City fixMid = init[i] as City;
                    City fixNext = init[(i + 1) % Cities.Length] as City;
                    City swapPrev;
                    City swapMid;
                    City swapNext;
                    int swapIndex;
                    do
                    {
                        swapIndex = m.Next(1,Cities.Length - 1);
                        swapPrev = init[swapIndex - 1 % Cities.Length] as City;
                        swapMid = init[swapIndex % Cities.Length] as City;
                        swapNext = init[swapIndex + 1 % Cities.Length] as City;
                    } while (fixPrev.costToGetTo(swapMid) == double.PositiveInfinity ||
                    swapMid.costToGetTo(fixNext) == double.PositiveInfinity ||
                    swapPrev.costToGetTo(fixMid) == double.PositiveInfinity ||
                    fixMid.costToGetTo(swapNext) == double.PositiveInfinity);

                    // found an agreeable swap, fix.
                    init[i] = swapMid;
                    init[swapIndex] = fixNext;
                }
                TSPSolution tmp = new TSPSolution(init);
                if (tmp.costOfRoute() < bssf.costOfRoute())
                    bssf = tmp;
            }
        }

        public void random()
        {
            // start our timer
            System.Diagnostics.Stopwatch timer = new System.Diagnostics.Stopwatch();
            timer.Start();

            Route = new ArrayList();
            // this is the trivial solution. 
            for (int x = 0; x < Cities.Length; x++)
            {
                Route.Add(Cities[x]);
            }
            // call this the best solution so far.  bssf is the route that will be drawn by the Draw method. 
            bssf = new TSPSolution(Route);
            
            Program.MainForm.tbCostOfTour.Text = "" + bssf.costOfRoute();
            Program.MainForm.tbElapsedTime.Text = Convert.ToString(timer.Elapsed);
        }


        public void greedy()
        {
            // start our timer
            System.Diagnostics.Stopwatch timer = new System.Diagnostics.Stopwatch();
            timer.Start();

            // get cost from nearest neighbor
            Random rand = new Random();
            double totalCost = double.PositiveInfinity;
            int start = 0;
            HashSet<int> visitedCities = new HashSet<int>() { start };
            ArrayList path = new ArrayList();

            while (double.IsInfinity(totalCost) || double.IsNaN(totalCost))
            {
                totalCost = 0;
                start = rand.Next(Cities.Length);
                visitedCities = new HashSet<int>() { start };
                path = new ArrayList() { Cities[start] };

                int current = start;
                do
                {
                    int bestCity = -1;
                    double bestCost = Double.PositiveInfinity;
                    for (int i = 0; i < Cities.Length; i++)
                    {
                        if (!visitedCities.Contains(i))
                        {
                            double cost = Cities[current].costToGetTo(Cities[i]);
                            if (cost < bestCost)
                            {
                                bestCity = i;
                                bestCost = cost;
                            }
                        }
                    }

                    totalCost += bestCost;
                    visitedCities.Add(bestCity);
                    path.Add(Cities[bestCity]);
                    current = bestCity;
                } while (visitedCities.Count != Cities.Length);
                totalCost += Cities[current].costToGetTo(Cities[start]);
            }

            bssf = new TSPSolution(path);

            timer.Stop();

            Program.MainForm.tbCostOfTour.Text = "" + bssf.costOfRoute();
            Program.MainForm.tbElapsedTime.Text = Convert.ToString(timer.Elapsed);

        }

        public void branchAndBound()
        {
            // start our timer
            System.Diagnostics.Stopwatch timer = new System.Diagnostics.Stopwatch();
            timer.Start();

            int solNumber = 0;
            int statesCreated = 0;
            int branchesConsidered = 0;
            int statesStored = 0;
            int statesPruned = 0;

            // generate bssf
            initBSSF();
            double bsf = costOfBssf();
            double initial = bsf;

            // generate initial cost matrix
            state s = new state(Cities.Length);
            for (int i = 0; i < Cities.Length; i++)
            {
                for (int j = 0; j < Cities.Length; j++)
                {
                    if (i == j)
                        s.costMatrix[i, j] = double.PositiveInfinity;
                    else
                        s.costMatrix[i, j] = Cities[i].costToGetTo(Cities[j]);
                }
            }
            reduce(s);

            // create priority queue and init with initial state
            PriorityQueue<state> pq = new PriorityQueue<state>();
            pq.Enqueue(s,0);

            // loop - while !empty and cost < bssf{   
            while (pq.Count > 0)
            {
                state cur = pq.Dequeue();
                if (cur.cost > bsf)
                {
                    statesPruned++;
                    continue;
                }

                state nextInc = cur;
                state nextEx;
                int cityOut = -1;
                int cityIn = -1;
                double improvement = double.NegativeInfinity;

                // for all edges
                for (int i = 0; i < cur.numEdges; i++)
                {
                    for (int j = 0; j < cur.numEdges; j++)
                    {
                        // if edge is 0
                        if (cur.costMatrix[i, j] == 0)
                        {
                            // calculate best if included
                            state tmp = includeEdge(cur, i, j);
                            branchesConsidered++;

                            // and improvement over exclude
                            double tmpImprov = worstCase(cur, i, j) - tmp.cost;

                            // and if keep if best improvement so far
                            if (improvement < tmpImprov)
                            {
                                nextInc = tmp;
                                cityOut = i;
                                cityIn = j;
                                improvement = tmpImprov;
                            }
                        }
                    }
                }

                if (nextInc.cost < bsf)
                {
                    // is this state a complete solution?
                    if (nextInc.numAddedEdges == nextInc.numEdges)
                    {
                        // transform into bssf
                        ArrayList route = new ArrayList();
                        int city = 0;
                        do
                        {
                            route.Add(Cities[city]);
                            city = nextInc.outEdges[city];
                        } while (city != 0);

                        // update
                        bssf = new TSPSolution(route);
                        bsf = costOfBssf();
                        solNumber++;
                    }
                    else
                    {
                        // we've found the state with the best improvement
                        // so calculate create the exclude state;
                        nextEx = new state(cur);
                        nextEx.costMatrix[cityOut, cityIn] = double.PositiveInfinity;
                        reduce(nextEx);

                        // enqueue both of the new states
                        pq.Enqueue(nextInc, Convert.ToInt32(nextInc.cost / (nextInc.numAddedEdges + 1)));
                        statesCreated += 2;

                        // enqueue if not infinite
                        if (nextEx.cost < bsf)
                            pq.Enqueue(nextEx, Convert.ToInt32(nextEx.cost / (nextInc.numAddedEdges + 1)));
                        else
                            statesPruned++;

                        // die with soemthing if we never actually expanded the state
                        if (nextInc == cur) throw new NotSupportedException();
                    }
                    if (pq.Count > statesStored)
                    {
                        statesStored = pq.Count;
                    }
                }
            } // end while loop

            timer.Stop();

            Program.MainForm.tbCostOfTour.Text = " " + bssf.costOfRoute();
            Program.MainForm.tbElapsedTime.Text = Convert.ToString(timer.Elapsed);
            Program.MainForm.tbStateInfo.Text = Convert.ToString(statesCreated) + "-" + Convert.ToString(statesStored) + "-" + Convert.ToString(statesPruned);
            Program.MainForm.tbInitialBSF.Text = Convert.ToString(initial);
            Program.MainForm.tbSolutionNum.Text = Convert.ToString(solNumber);
            // do a refresh. 
            Program.MainForm.Invalidate();
        } // end function   

        private class Ant
        {
            public HashSet<int> unvisited;
            public int startCity;
            public int curCity;
            public int prevCity;
            public int[] tour;
            public double curCost;

            public Ant(int cities, int start)
            {
                // initialize agents
                tour = new int[cities];
                startCity = start;
                unvisited = new HashSet<int>();
                for (int i = 0; i < cities; i++)
                {
                    if (i != start)
                        unvisited.Add(i);
                }
                curCity = start;
                prevCity = -1;
                curCost = 0;
            }
        }

        public void groupTSP()
        {
            // start our timer
            System.Diagnostics.Stopwatch timer = new System.Diagnostics.Stopwatch();
            timer.Start();

            // constants used in heuristics
            const int ITERATIONS = 50;
            const double Q0 = 0.9;
            const double BETA = 50.0;
            const double ALPHA = 0.1;

            // get cost from nearest neighbor, used in our heuristic
            greedy();

            double nnCost = costOfBssf();

            // our fastest ant through all the iterations
            Ant bestAnt = null;
            double bestCost = Double.PositiveInfinity;

            // initialize pheramone level matrix
            double[,] pheromoneLevels = new double[Cities.Length, Cities.Length];
            for (int i = 0; i < Cities.Length; i++)
            {
                for (int j = 0; j < Cities.Length; j++)
                {
                    pheromoneLevels[i, j] = 1.0 / Cities.Length;
                }
            }

            // start iterations
            for (int h = 0; h < ITERATIONS; h++)
            {
                // our fastest ant in the iteration
                Ant bestAntIter = null;
                double bestCostIter = Double.PositiveInfinity;

                // initialize ants
                List<Ant> ants = new List<Ant>();
                for (int i = 0; i < Cities.Length; i++)
                {
                    ants.Add(new Ant(Cities.Length, i));
                }

                // iterate through each time step
                for (int i = 0; i < Cities.Length - 1; i++)
                {
                    // determine each ants next move
                    List<Ant> remainingAnts = new List<Ant>(ants);
                    foreach (Ant ant in remainingAnts)
                    {
                        int nextCity = -1;
                        double nextDistance = Double.PositiveInfinity;

                        Random random = new Random();
                        double q = random.NextDouble();
                        if (q < Q0)
                        {
                            // select next city based on distance / pheramone only
                            double bestScore = Double.NegativeInfinity;
                            foreach (int city in ant.unvisited)
                            {
                                double distance = Cities[ant.curCity].costToGetTo(Cities[city]);
                                double score = pheromoneLevels[ant.curCity, city] * Math.Pow(1 / distance, BETA);
                                if (score > bestScore)
                                {
                                    nextCity = city;
                                    bestScore = score;
                                    nextDistance = distance;
                                }
                            }
                        }
                        else
                        {
                            // select city randomly with different weights given to each city based on distance / pheramone
                            double runningTotal = 0.0;
                            double randomValue = random.NextDouble();

                            // get sum of scores
                            double summedScores = 0.0;
                            foreach (int city in ant.unvisited)
                            {
                                double distance = Cities[ant.curCity].costToGetTo(Cities[city]);
                                summedScores += pheromoneLevels[ant.curCity, city] * Math.Pow(1 / distance, BETA);
                            }

                            // determine probability for each city
                            foreach (int city in ant.unvisited)
                            {
                                double distance = Cities[ant.curCity].costToGetTo(Cities[city]);
                                runningTotal += (pheromoneLevels[ant.curCity, city] * Math.Pow(1 / distance, BETA)) / summedScores;
                                if (runningTotal > randomValue)
                                {
                                    nextCity = city;
                                    nextDistance = distance;
                                    break;
                                }
                            }
                        }

                        if (nextCity != -1)
                        {
                            // apply local trail updating
                            pheromoneLevels[ant.curCity, nextCity] += (1 - ALPHA) * pheromoneLevels[ant.curCity, nextCity] + ALPHA * (1 / (Cities.Length * nnCost));

                            // add the picked city to the tour
                            ant.tour[ant.curCity] = nextCity;
                            ant.unvisited.Remove(nextCity);
                            ant.curCost += nextDistance;
                            ant.prevCity = ant.curCity;
                            ant.curCity = nextCity;
                        }
                        else
                        {
                            //deal with stuck ant
                            ants.Remove(ant);
                        }
                    }
                }

                // finalize tours and find best ant
                foreach (Ant ant in ants)
                {
                    // finalize tour
                    ant.tour[ant.curCity] = ant.startCity;
                    ant.curCost += Cities[ant.curCity].costToGetTo(Cities[ant.startCity]);

                    // check against current best cost
                    if (ant.curCost < bestCostIter)
                    {
                        bestAntIter = ant;
                        bestCostIter = ant.curCost;
                    }
                }

                if (bestAntIter != null)
                {
                    // apply global trail updating
                    int current = bestAntIter.startCity;
                    do
	                {
                        pheromoneLevels[current, bestAntIter.tour[current]] += (1 - ALPHA) * pheromoneLevels[current, bestAntIter.tour[current]] + ALPHA * (1 / bestCostIter);
                        current = bestAntIter.tour[current];
	                } while (current != bestAntIter.startCity);

                    // see if it is the best ant through all iterations so far
                    if (bestCostIter < bestCost)
                    {
                        bestAnt = bestAntIter;
                        bestCost = bestCostIter;
                    }
                }
            }

            if (bestAnt != null)
            {
                ArrayList route = new ArrayList();
                int curCity = bestAnt.startCity;
                do
                {
                    route.Add(Cities[curCity]);
                    curCity = bestAnt.tour[curCity];
                } while (curCity != bestAnt.startCity);
                bssf = new TSPSolution(route);
            }

            timer.Stop();

            Program.MainForm.tbCostOfTour.Text = " " + bssf.costOfRoute();
            Program.MainForm.tbElapsedTime.Text = Convert.ToString(timer.Elapsed);

            // do a refresh. 
            Program.MainForm.Invalidate();
        }

    }

}
