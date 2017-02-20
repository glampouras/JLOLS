/* 
 * Copyright (C) 2016 Gerasimos Lampouras
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package jarow;

import java.util.Random;
import java.util.logging.Logger;



public class RandomGaussian {
  
    /**
     *
     * @param aArgs
     */
    public static void main(String... aArgs){
    RandomGaussian gaussian = new RandomGaussian();
    double MEAN = 100.0f; 
    double VARIANCE = 5.0f;
    for (int idx = 1; idx <= 10; ++idx){
      log("Generated : " + gaussian.getGaussian(MEAN, VARIANCE));
    }
  }
    
  private final Random randomGen = new Random();
  
    /**
     *
     * @param aMean
     * @param aVariance
     * @return
     */
    public double getGaussian(double aMean, double aVariance){
    return aMean + randomGen.nextGaussian() * aVariance;
  }

  private static void log(Object aMsg){
  }
    private static final Logger LOG = Logger.getLogger(RandomGaussian.class.getName());
} 