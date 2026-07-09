
#pragma once

#include "model/perception/text.hpp"
#include "model/cognition/memory.hpp"
#include "model/cognition/world_model.hpp"
#include "model/expression/text.hpp"


namespace model
{
    
class model
{
private:
    //  perception
    perception::text text_perception;

    //  cognition
    memory memory;
    world_model brain;

    //  expression
    text_decoder t_decoder;

public:
    model();

    void load();
    void save();
    void create();

    void sense_text(perception::input input);

};


}

