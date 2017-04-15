function Update_D(Solver_D_, Solver_D, batch_D, gt_D, iter_D)

       active_D = Solver_D_.net.forward(batch_D);    
       delta_D = cell(size(active_D));
       for c = 1:length(active_D)
           active_D_ = active_D{c};
           delta_D{c} = zeros(size(active_D{c}));
           [delta_D_, loss_D] = GFC_BCE_loss1(active_D_, gt_D, 'train');
           Solver_D.loss(iter_D) = loss_D;          
           delta_D{c} = delta_D_;
       end
      
       fprintf('loss_D = %i', loss_D); fprintf('  ');
      
       if ~isnan(Solver_D.loss(iter_D))
              f = Solver_D_.net.backward(delta_D);
              Solver_D_.update();
       else
           error('NAN');
       end
       
       if mod(iter_D, 100) == 0
           Solver_D_.net.save(save_file_D);
       end
end