using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace TSP
{
    class Matrix : IComparable<Matrix>
    {
        private double[,] cities;
        private double minCost;
        private int[] entered;
        private int[] exited;
        private double averageCostPerEdge;

        public Matrix(City[] newCities)
        {
            cities = new double[newCities.Length, newCities.Length];
            for(int i = 0; i < newCities.Length; i++)
            {
                for(int j = 0; j < newCities.Length; j++)
                {
                    cities[i,j] = newCities[i].costToGetTo(newCities[j]);
                }
            }


            entered = new int[newCities.Length];
            exited = new int[newCities.Length];
            for (int i = 0; i < newCities.Length; i++)
            {
                entered[i] = -1;
                exited[i] = -1;
            }


            removePathsToSelf();
            minCost = MinimizeMatrix();
        }

        public Matrix(Matrix other)
        {
            cities = (double[,])other.cities.Clone();
            minCost = other.minCost;
            entered = new int[other.entered.Length];
            exited = new int[other.exited.Length];

            for (int i = 0; i < other.entered.Length; i++)
            {
                entered[i] = other.entered[i];
                exited[i] = other.exited[i];
            }
        }


        private void removePathsToSelf()
        {
            for (int i = 0; i < getMatrixRowLength(); i++)
            {
                cities[i, i] = double.PositiveInfinity;
            }
        }

        /**
         * This returns zero if they have the same weighted cost
         * Otherwise it returns a positive number if n1 > n2
         * OR a negative number if n2 > n1. To get the weighted cost
         * we add to their base costs the average weight of their remaining paths
         * multiplied by how many moves they have yet to make.
         */
        public int CompareTo(Matrix other)
        {
            double thisWeightedCost = minCost + averageCostPerEdge * getNumberOfCitiesLeftToVisit();
            double otherWeightedCost = other.minCost + other.averageCostPerEdge * other.getNumberOfCitiesLeftToVisit();
            return thisWeightedCost.CompareTo(otherWeightedCost);
        }

        public double getMinCost()
        {
            return minCost;
        }

        public int getMatrixRowLength()
        {
            return getNumberOfRows();
        }

        public bool hasExited(int city)
        {
            return exited[city] != -1;
        }

        public bool hasEntered(int city)
        {
            return entered[city] != -1;
        }

        public void enter(int fromCity, int destCity)
        {
            entered[destCity] = fromCity;
        }

        public void exit(int fromCity, int destCity)
        {
            exited[fromCity] = destCity;
        }

        public double getCost(int row, int col)
        {
            return cities[row, col];
        }

        public int[] getCitiesExited()
        {
            return exited;
        }

        public int[] getCitiesEntered()
        {
            return entered;
        }

        public int getNumberOfCitiesLeftToVisit()
        {
            int numberLeftToVisit = 0;
            for (int i = 0; i < entered.Length; i++)
            {
                if (entered[i] == -1)
                    numberLeftToVisit++;
            }

            return numberLeftToVisit;
        }

        public int getNumberOfCitiesVisited()
        {
            int numberVisited = 0;
            for (int i = 0; i < entered.Length; i++)
            {
                if (entered[i] != -1)
                    numberVisited++;
            }

            return numberVisited;
        }

        private int getNumberOfRows()
        {
            return cities.GetLength(0);         
        }

        public double MinimizeMatrix()
        {
            double resultCostAddition = 0;
            //for each row
            for (int i = 0; i < getNumberOfRows(); i++)
            {
                //find the min value for the row
                double min = cities[i,0];
                for (int j = 0; j < getNumberOfRows() && min != 0; j++)
                {
                    if (cities[i, j] < min &&
                        !double.IsInfinity(cities[i, j]) &&
                        !double.IsNaN(cities[i, j]))
                    {
                        min = cities[i, j];
                    }
                }

                if ((double.IsInfinity(min) || double.IsNaN(min)) && entered[i] == -1 && exited[i] == -1)
                {
                    return double.PositiveInfinity;
                }

                //update the row
                if (!(double.IsInfinity(min) || double.IsNaN(min)) && min != 0)
                {
                    resultCostAddition += min;
                    for (int j = 0; j < getNumberOfRows() && min != 0; j++)
                    {
                        cities[i, j] = cities[i, j] - min;
                    }
                }
            }

            //for each column
            for (int i = 0; i < getNumberOfRows(); i++)
            {
                //find the min value for the column
                double min = cities[0, i];
                for (int j = 0; j < getNumberOfRows() && min != 0; j++)
                {
                    if (cities[j, i] < min &&
                        !double.IsInfinity(cities[j, i]) &&
                        !double.IsNaN(cities[j, i]))
                    {
                        min = cities[j, i];
                    }
                }

                if ((double.IsInfinity(min) || double.IsNaN(min)) && entered[i] == -1 && exited[i] == -1)
                {
                    return double.PositiveInfinity;
                }

                //update the row
                if (!(double.IsInfinity(min) || double.IsNaN(min)) && min != 0)
                {
                    resultCostAddition += min;
                    for (int j = 0; j < getNumberOfRows() && min != 0; j++)
                    {
                        cities[j, i] = cities[j, i] - min;
                    }
                }
            }

            averageCostPerEdge = getAverageCostPerEdge();

            minCost += resultCostAddition;
            return minCost;
        }

        public double getAverageCostPerEdge()
        {
            int numberOfEdges = 0;
            double totalCostOfEdges = 0;
            for (int i = 0; i < getNumberOfRows(); i++)
            {
                for (int j = 0; j < getNumberOfRows(); j++)
                {
                    if (!double.IsInfinity(cities[i, j]) && !double.IsNaN(cities[i, j]))
                    {
                        numberOfEdges++;
                        totalCostOfEdges += cities[i,j];
                    }
                }
            }

            return (totalCostOfEdges / numberOfEdges);
        }

        public double getExcludeCost(int row, int col)
        {
            double result = minCost;

            double curMin = double.PositiveInfinity;

            //Get the min value from the column not including the current cell
            for (int i = 0; i < getNumberOfRows(); i++)
            {
                if (!double.IsInfinity(cities[i,col]) && cities[i,col] < curMin && i != row)
                {
                    curMin = cities[i,col];
                }
            }

            result += curMin;
            curMin = double.PositiveInfinity;
            //get the min value from the row not including the current cell
            for (int i = 0; i < getNumberOfRows(); i++)
            {
                if (!double.IsInfinity(cities[row, i]) && cities[row, i] < curMin && i != col)
                {
                    curMin = cities[row,i];
                }
            }

            result += curMin;

            return result;
        }

        public double getIncludeCost(int row, int col)
        {
            double result = minCost + cities[row, col];
            HashSet<KeyValuePair<int, int>> minimizedCells = new HashSet<KeyValuePair<int,int>>();
            KeyValuePair<int, int> curMin;


            if(double.IsInfinity(cities[row,col]))
                return double.PositiveInfinity;

            //Get the min value from each column not including the current cell's column
            for (int i = 0; i < getNumberOfRows(); i++)
            {
                if (entered[i] == -1 && i != col)
                {
                    curMin = new KeyValuePair<int, int>(0, i);
                    for (int j = 0; j < getNumberOfRows(); j++)
                    {
                        if (!double.IsInfinity(cities[j, i]) &&
                            !double.IsNaN(cities[j, i]) &&
                            cities[j, i] < cities[curMin.Key, curMin.Value])
                        {
                            curMin = new KeyValuePair<int, int>(j, i);
                        }
                    }

                    if (double.IsInfinity(cities[curMin.Key, curMin.Value]) ||
                        double.IsNaN(cities[curMin.Key, curMin.Value]))
                        return double.PositiveInfinity;
                    else
                    {
                        result += cities[curMin.Key, curMin.Value];
                        minimizedCells.Add(curMin);
                    }
                }
                
            }

            //get the min value from the row not including the current cell
            for (int i = 0; i < getNumberOfRows(); i++)
            {
                if (exited[i] == -1 && i != row)
                {
                    curMin = new KeyValuePair<int, int>(i, 0);
                    for (int j = 0; j < getNumberOfRows(); j++)
                    {
                        if (!double.IsInfinity(cities[i, j]) &&
                            !double.IsNaN(cities[i, j]) &&
                            cities[i, j] < cities[curMin.Key, curMin.Value])
                        {
                            curMin = new KeyValuePair<int, int>(i, j);
                        }
                    }

                    if (double.IsInfinity(cities[curMin.Key, curMin.Value]) &&
                        double.IsNaN(cities[curMin.Key, curMin.Value]))
                        return double.PositiveInfinity;
                    else
                        result += cities[curMin.Key, curMin.Value];
                }
            }

            return result;
        }


        public void Include(int row, int col)
        {

            enter(row, col);
            exit(row, col);
            minCost += cities[row, col];

            for (int i = 0; i < getNumberOfRows(); i++)
            {
                cities[i, col] = double.PositiveInfinity;
                cities[row, i] = double.PositiveInfinity;
            }

            //we remove the backward edge
            cities[col, row] = double.PositiveInfinity;

            removePrematureLoops(row, col);

            minCost = MinimizeMatrix();

        }

        public void removePrematureLoops(int row, int col)
        {
            //The newly entered city cannot go to any of the previously exited states
            //we skip past the first entry in the arraylist which is a -1
            int start = row; 
            int end = col;
            
            // The new edge may be part of a partial solution.  Go to the end of that solution.  
            while (exited[end] != -1)
            {
                end = exited[end];
            }

            // Similarly, go to the start of the new partial solution.  
            while (entered[start] != -1) 
            {
                start = entered[start];
            }

            // Delete the edges that would make partial cycles, unless we’re ready to finish the tour  
            if (getNumberOfCitiesLeftToVisit() > 0)
            {
                while (start != row)
                {
                    cities[end, start] = double.PositiveInfinity;
                    cities[row, start] = double.PositiveInfinity;
                    start = exited[start];
                }
            }
        }

        public void Exclude(int row, int col)
        {
            cities[row, col] = double.PositiveInfinity;

            minCost = MinimizeMatrix();
        }

        public int getNextRow()
        {
            int exit = -1;

            for (int i = 0; i < exited.Length; i++)
            {
                if (exited[i] != -1)
                {
                    exit = i;
                    break;
                }
            }

            if (exit == -1)
                return 0;

            while (exited[exit] != -1)
                exit = exited[exit];


            return exit;


        }

    }
}
