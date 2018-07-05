#include <vector>
#include <set>
#include <iostream>
#include <stdlib.h>
#include "Eigen.h"
#include <tuple>
using namespace std;
using namespace Eigen;

namespace NSAggregation {
  typedef tuple<Vector3f,Vector2f> Match;
  typedef vector<Match> Frame; // a list of keypoints form a frame
  typedef tuple<Vector2f,int> Tag; // a point with a frame index forms a tag
  typedef tuple<Vector3f, vector<Tag> > OneToManyMap; // a 3d point with a list of tags is the OTMM
  typedef vector<OneToManyMap> Aggregation; // a list of OTMM forms the Aggregation
}

using namespace NSAggregation;
class Aggregator {
public:
  Aggregator(float dist = 0.01) : m_matching_distance(dist)
  {}
  // bunch of debugging functions here
  /*
    What's going on:
      - Select `img` random 3d points
      - for each dummy frame, select 1 random 2d point and match it with a 3d point
      - *** except for the middle match, which is deliberately set to matching the first 3d point
          so that the aggregator has something to aggregate ***
      - gives the result back.
  */
  static vector<Frame> generateDummyData(int img = 5) {
    vector<Frame> frames;
    vector<Vector3f> groundTruth;
    // select 3d points
    for(int i = 0; i < img; i++) groundTruth.push_back(Vector3f::Random());
    for(int i = 0; i < img; i++) {
      Frame frame;
      for(int j = 0; j < img; j++) {
        Match match = make_tuple(groundTruth[(j == img/2)?0:j],Vector2f::Random());
        frame.push_back(match);
      }
      frames.push_back(frame);
    }
    return frames;
  }
  static void printMatch(Match match) {
    cout << "Match:" << get<0>(match).transpose() << "->" << get<1>(match).transpose() << endl;
  }
  static void printFrame(Frame frame) {
    cout << "Frame:" << endl;
    for(Match& match: frame) {
      cout << '\t';
      printMatch(match);
    }
    cout << endl;
  }
  static void printTag(Tag tag) {
    cout << "Tag:" << get<0>(tag).transpose() << "|" << get<1>(tag) << endl;
  }
  static void printOneToManyMap(OneToManyMap otmm) {
    cout << "Mapping from:" << get<0>(otmm).transpose() << endl;
    for(Tag& t: get<1>(otmm)) {
      cout << "\t\t";
      printTag(t);

    }
  }
  static void printAggregation(const Aggregation& agg) {
    cout << "Aggregation:" << endl;
    for(const OneToManyMap& otmm : agg) {
      printOneToManyMap(otmm);
    }
  }
  // functional methods
  void initialize(const vector<Frame> & frames) {
    m_aggregation.clear();
    m_frames.clear();
    m_frames = vector<Frame>(frames);
    constructAggregation();
  }

  const Aggregation& getAggregation() const {
    return m_aggregation;
  }

 vector<Vector3f> get3DPoints() {
    vector<Vector3f> points;
    for(const OneToManyMap& otmm: m_aggregation) {
      points.push_back(get<0>(otmm));
    }
    return points;
  }
private:
  // bunch of helper functions here
  /*
    What it does: Turns a group of frames (m_frames) into aggregations (m_aggregation)
    How:
      - For each frame f...
        - For each match m...
          - tag the 2d points to the index of f
          - get the 3d point from m
          - get the closest 3d point (within the threshold)
          - if theres no such point...
            - add this 3d point to the aggregation
            - add the tag to this 3d point
          - else...
              add the tag to this "closes" 3d point
  */
  void constructAggregation() {
    m_aggregation.clear();
    for(int f = 0; f < m_frames.size(); f++) {
      for(Match& m: m_frames[f]) {
        // turn the 2d point to a tag
        Tag tag = make_tuple(get<1>(m),f);
        // get the 3d point
        Vector3f p = get<0>(m);
        // get the closes3d point
        int ind = indClosestPoint(get3DPoints(),p);
        if(ind == -1) {
          // no matches found, sorry
          // create a OTMM
          vector<Tag> tags;
          tags.push_back(tag);
          OneToManyMap otmm = make_tuple(p,tags);
          m_aggregation.push_back(otmm);
        } else {
          OneToManyMap& otmm = m_aggregation.at(ind); // yes i need a reference thanks...
          get<1>(otmm).push_back(tag);
        }
      }
    }
  }
  float pointDistance(const Vector3f& a, const Vector3f& b) {
    return (a - b).norm();
  }
  bool is3DPointsMatch(const Vector3f& a, const Vector3f& b) {
    return pointDistance(a,b) <= m_matching_distance;
  }
  // return the "closest point" amoungst the aggregated points
  int indClosestPoint(const vector<Vector3f>& points, const Vector3f& p) {
    int ind = -1;
    float curDist = -1;
    for(int i = 0; i < points.size(); i++) {
      const Vector3f& a = points[i];
      float dist = pointDistance(a,p);
      if(curDist == -1 || dist < curDist) {
        ind = i;
        curDist = dist;
      }
    }
    // final check here: is the closest point still not as close as the matching distance?
    return (curDist <= m_matching_distance)?ind:-1;
  }

  float m_matching_distance;
  vector<Frame> m_frames;
  Aggregation m_aggregation;
};
