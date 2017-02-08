/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package jdagger;

/**
 *
 * @author Gerasimos Lampouras
 */
import java.util.ArrayList;
import org.apache.commons.collections4.MapIterator;
import org.apache.commons.collections4.map.LRUMap;

/**
 *
 * @author Gerasimos Lampouras
 * @param <K>
 * @param <T>
 */
public class WordSequenceCache<K, T> {
 
    private long timeToLive;
    private LRUMap cacheMap;
 
    /**
     *
     */
    protected class CacheObject {

        /**
         *
         */
        public long lastAccessed = System.currentTimeMillis();

        /**
         *
         */
        public T value;
 
        /**
         *
         * @param value
         */
        protected CacheObject(T value) {
            this.value = value;
        }
    }
 
    /**
     *
     * @param timeToLive
     * @param timerInterval
     * @param maxItems
     */
    public WordSequenceCache(long timeToLive, final long timerInterval, int maxItems) {
        this.timeToLive = timeToLive * 1000;
 
        cacheMap = new LRUMap(maxItems);
 
        if (timeToLive > 0 && timerInterval > 0) {
 
            Thread t = new Thread(new Runnable() {
                public void run() {
                    while (true) {
                        try {
                            Thread.sleep(timerInterval * 1000);
                        } catch (InterruptedException ex) {
                        }
                        cleanup();
                    }
                }
            });
 
            t.setDaemon(true);
            t.start();
        }
    }
 
    /**
     *
     * @param key
     * @param value
     */
    public void put(K key, T value) {
        synchronized (cacheMap) {
            cacheMap.put(key, new CacheObject(value));
        }
    }
    
    /**
     *
     */
    public void keyset() {
        synchronized (cacheMap) {
            System.out.println(cacheMap.keySet());
        }
    }
 
    /**
     *
     * @param key
     * @return
     */
    @SuppressWarnings("unchecked")
    public T get(K key) {
        synchronized (cacheMap) {
            CacheObject c = (CacheObject) cacheMap.get(key);
 
            if (c == null)
                return null;
            else {
                c.lastAccessed = System.currentTimeMillis();
                return c.value;
            }
        }
    }
 
    /**
     *
     * @param key
     */
    public void remove(K key) {
        synchronized (cacheMap) {
            cacheMap.remove(key);
        }
    }
 
    /**
     *
     * @return
     */
    public int size() {
        synchronized (cacheMap) {
            return cacheMap.size();
        }
    }
 
    /**
     *
     */
    @SuppressWarnings("unchecked")
    public void cleanup() {
 
        long now = System.currentTimeMillis();
        ArrayList<K> deleteKey = null;
 
        synchronized (cacheMap) {
            MapIterator itr = cacheMap.mapIterator();
 
            deleteKey = new ArrayList<K>((cacheMap.size() / 2) + 1);
            K key = null;
            CacheObject c = null;
 
            while (itr.hasNext()) {
                key = (K) itr.next();
                c = (CacheObject) itr.getValue();
 
                if (c != null && (now > (timeToLive + c.lastAccessed))) {
                    deleteKey.add(key);
                }
            }
        }
 
        for (K key : deleteKey) {
            synchronized (cacheMap) {
                cacheMap.remove(key);
            }
 
            Thread.yield();
        }
    }
}
