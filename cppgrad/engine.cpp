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

public:
    Value(
        double data,
        set<shared_ptr<Value>> children = {},
        string op = "",
        string label = "") : data(data),
                             _prev(move(children)),
                             _operation(move(op)),
                             _label(move(label)) {}

    string repr() const
    {
        return "Value(data)= " + to_string(data);
    }
};

ostream &operator<<(ostream &os, const Value &v)
{
    return os << v.repr();
}