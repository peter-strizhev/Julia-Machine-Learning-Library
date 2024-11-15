using Test
using ..MySimpleNN

# Define test cases
@testset "Neural Network Tests" begin
    # Test initializing the network
    model = initialize_network([3, 5, 2], [relu, sigmoid])
    @test length(model.layers) == 2
    @test size(model.layers[1]) == (5, 3)
    @test size(model.layers[2]) == (2, 5)
end
