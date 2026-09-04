#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <vector>

using namespace std;

class Value
{
public:
    double data;
    double grad = 0.0;

private:
    function<void()> _backward = []() {};
    set<shared_ptr<Value>> _prev;
    string _operation;
    string _label;
};