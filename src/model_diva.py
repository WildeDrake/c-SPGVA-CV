import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributions as dist

'''
Módulo | Rol                    | Tipo        | Entrada                | Salida
qzd    | Encoder de sujeto      | Inferencial | x                      | zd
qzx    | Encoder residual       | Inferencial | x                      | zx
qzy    | Encoder de gesto       | Inferencial | x                      | zy
px     | Decoder                | Generativo  | zd, zx, zy, c          | x_recon
pzd    | Generador latente      | Generativo  | d (one-hot del sujeto) | zd
pzy    | Generador latente      | Generativo  | y (one-hot del gesto)  | zy
qd     | Clasificador auxiliar  | Auxiliar    | zd                     | d_hat
qy     | Clasificador auxiliar  | Auxiliar    | zy                     | y_hat
'''


#------------------------------ Encoders  ------------------------------#
# Encoder de sujeto
class qzd(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim):
        super(qzd, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 12, kernel_size=(5, 13), stride=1, bias=False), nn.BatchNorm2d(12), nn.ReLU(),
            nn.MaxPool2d((1, 2)), 
            nn.Conv2d(12, 24, kernel_size=(4, 11), stride=1, bias=False), nn.BatchNorm2d(24), nn.ReLU(),
        )
        self.fc11 = nn.Sequential(nn.Linear(240, zd_dim))
        self.fc12 = nn.Sequential(nn.Linear(240, zd_dim), nn.Softplus())
        torch.nn.init.xavier_uniform_(self.encoder[0].weight)
        torch.nn.init.xavier_uniform_(self.encoder[4].weight)
        torch.nn.init.xavier_uniform_(self.fc11[0].weight)
        self.fc11[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc12[0].weight)
        self.fc12[0].bias.data.zero_()

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(-1, 24 * 1 * 10)
        zd_loc = self.fc11(h)
        zd_scale = self.fc12(h) + 1e-7
        return zd_loc, zd_scale

# Encoder residual
class qzx(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim):
        super(qzx, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 12, kernel_size=(5, 13), stride=1, bias=False), nn.BatchNorm2d(12), nn.ReLU(),
            nn.MaxPool2d((1, 2)),
            nn.Conv2d(12, 24, kernel_size=(4, 11), stride=1, bias=False), nn.BatchNorm2d(24), nn.ReLU(),
        )
        self.fc11 = nn.Sequential(nn.Linear(240, zx_dim))
        self.fc12 = nn.Sequential(nn.Linear(240, zx_dim), nn.Softplus())
        torch.nn.init.xavier_uniform_(self.encoder[0].weight)
        torch.nn.init.xavier_uniform_(self.encoder[4].weight)
        torch.nn.init.xavier_uniform_(self.fc11[0].weight)
        self.fc11[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc12[0].weight)
        self.fc12[0].bias.data.zero_()

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(-1, 24 * 1 * 10)
        zx_loc = self.fc11(h)
        zx_scale = self.fc12(h) + 1e-7
        return zx_loc, zx_scale

# Encoder de gesto
class qzy(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim):
        super(qzy, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 12, kernel_size=(5, 13), stride=1, bias=False), nn.BatchNorm2d(12), nn.ReLU(),
            nn.MaxPool2d((1, 2)),
            nn.Conv2d(12, 24, kernel_size=(4, 11), stride=1, bias=False), nn.BatchNorm2d(24), nn.ReLU(),
        )
        self.fc11 = nn.Sequential(nn.Linear(240, zy_dim))
        self.fc12 = nn.Sequential(nn.Linear(240, zy_dim), nn.Softplus())
        torch.nn.init.xavier_uniform_(self.encoder[0].weight)
        torch.nn.init.xavier_uniform_(self.encoder[4].weight)
        torch.nn.init.xavier_uniform_(self.fc11[0].weight)
        self.fc11[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc12[0].weight)
        self.fc12[0].bias.data.zero_()
        
    def forward(self, x):
        h = self.encoder(x)
        h = h.view(-1, 240)
        zy_loc = self.fc11(h)
        zy_scale = self.fc12(h) + 1e-7
        return zy_loc, zy_scale


#------------------------------- Decoder  ------------------------------#

# Decoder Principal
class px(nn.Module):
    # Constructor modificado para aceptar c_dim
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim, c_dim): 
        super(px, self).__init__()
        self.input_dim = zd_dim + zx_dim + zy_dim + c_dim
        self.fc1 = nn.Sequential(nn.Linear(self.input_dim, 240, bias=False), nn.BatchNorm1d(240), nn.ReLU())
        self.de1 = nn.Sequential(nn.ConvTranspose2d(24, 36, kernel_size=(4, 11), stride=1, padding=0, bias=False),
                                 nn.BatchNorm2d(36), nn.ReLU())
        self.up2 = nn.Upsample([4, 40])
        self.de2 = nn.Sequential(nn.ConvTranspose2d(36, 48, kernel_size=(5, 13), stride=1, padding=0, bias=False),
                                 nn.BatchNorm2d(48), nn.ReLU())
        self.de3 = nn.Sequential(nn.Conv2d(48, 48, kernel_size=1, stride=1))

    # Forward modificado para aceptar c y concatenarlo
    def forward(self, zd, zx, zy, c):
        if zx is None:
            zdzxyc = torch.cat((zd, zy, c), dim=-1)
        else:
            zdzxyc = torch.cat((zd, zx, zy, c), dim=-1) 
        h = self.fc1(zdzxyc) 
        h = h.view(-1, 24, 1, 10)
        h = self.de1(h)
        h = self.up2(h)
        h = self.de2(h)
        loc_img = self.de3(h)
        return loc_img


#--------------------------- Generador latente ------------------------------#

# decoder de etiquetas de dominio de sujeto
class pzd(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim):
        super(pzd, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(d_dim, zd_dim, bias=False), nn.BatchNorm1d(zd_dim), nn.ReLU())
        self.fc21 = nn.Sequential(nn.Linear(zd_dim, zd_dim))
        self.fc22 = nn.Sequential(nn.Linear(zd_dim, zd_dim), nn.Softplus())
        torch.nn.init.xavier_uniform_(self.fc1[0].weight)
        torch.nn.init.xavier_uniform_(self.fc21[0].weight)
        self.fc21[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc22[0].weight)
        self.fc22[0].bias.data.zero_()

    def forward(self, d):
        hidden = self.fc1(d)
        zd_loc = self.fc21(hidden)
        zd_scale = self.fc22(hidden) + 1e-7
        return zd_loc, zd_scale

# Decoder de etiquetas de gesto
class pzy(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim):
        super(pzy, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(y_dim, zy_dim, bias=False), nn.BatchNorm1d(zy_dim), nn.ReLU())
        self.fc21 = nn.Sequential(nn.Linear(zy_dim, zy_dim))
        self.fc22 = nn.Sequential(nn.Linear(zy_dim, zy_dim), nn.Softplus())
        torch.nn.init.xavier_uniform_(self.fc1[0].weight)
        torch.nn.init.xavier_uniform_(self.fc21[0].weight)
        self.fc21[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc22[0].weight)
        self.fc22[0].bias.data.zero_()

    def forward(self, y):
        hidden = self.fc1(y)
        zy_loc = self.fc21(hidden)
        zy_scale = self.fc22(hidden) + 1e-7
        return zy_loc, zy_scale
    

#--------------------------- Classificadores Auxiliares -------------------------#

# Classificador de sujeto
class qd(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim):
        super(qd, self).__init__()
        self.fc1 = nn.Linear(zd_dim, d_dim)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.zero_()

    def forward(self, zd):
        h = F.relu(zd)
        loc_d = self.fc1(h)
        return loc_d

# Classificador de gesto
class qy(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim):
        super(qy, self).__init__()
        self.fc1 = nn.Linear(zy_dim, y_dim)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.zero_()

    def forward(self, zy):
        h = F.relu(zy)
        loc_y = self.fc1(h)
        return loc_y


#--------------------------- Modelo DIVA Completo ------------------------------#

class DIVA(nn.Module):
    # Constructor
    def __init__(self, args):
        super(DIVA, self).__init__()
        self.zd_dim = args.zd_dim 
        self.zx_dim = args.zx_dim 
        self.zy_dim = args.zy_dim 
        self.d_dim = args.d_dim 
        self.x_dim = args.x_dim 
        self.y_dim = args.y_dim 
        self.c_dim = args.c_dim 
        self.start_zx = self.zd_dim
        self.start_zy = self.zd_dim + self.zx_dim
        
        self.px = px(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim, self.c_dim)
        self.pzd = pzd(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim)
        self.pzy = pzy(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim)
        
        self.qzd = qzd(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim)
        if self.zx_dim != 0:
            self.qzx = qzx(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim)
        self.qzy = qzy(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim)
        
        self.qd = qd(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim)
        self.qy = qy(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim)
        
        self.aux_loss_multiplier_y = args.aux_loss_multiplier_y
        self.aux_loss_multiplier_d = args.aux_loss_multiplier_d
        self.beta_d = args.beta_d
        self.beta_x = args.beta_x
        self.beta_y = args.beta_y
        self.cuda()

    # Forward
    def forward(self, d, x, y, c):
        zd_q_loc, zd_q_scale = self.qzd(x)
        if self.zx_dim != 0:
            zx_q_loc, zx_q_scale = self.qzx(x)
        zy_q_loc, zy_q_scale = self.qzy(x)

        qzd = dist.Normal(zd_q_loc, zd_q_scale)
        zd_q = qzd.rsample()
        if self.zx_dim != 0:
            qzx = dist.Normal(zx_q_loc, zx_q_scale)
            zx_q = qzx.rsample()
        else:
            qzx = None
            zx_q = None
        qzy = dist.Normal(zy_q_loc, zy_q_scale)
        zy_q = qzy.rsample()

        x_recon = self.px(zd_q, zx_q, zy_q, c)
        
        zd_p_loc, zd_p_scale = self.pzd(d)
        if self.zx_dim != 0:
            # 🚨 CORRECCIÓN: Usar device de 'd' para consistencia
            zx_p_loc, zx_p_scale = torch.zeros(zd_p_loc.size()[0], self.zx_dim, device=d.device), \
            torch.ones(zd_p_loc.size()[0], self.zx_dim, device=d.device)
        zy_p_loc, zy_p_scale = self.pzy(y)

        pzd = dist.Normal(zd_p_loc, zd_p_scale)
        if self.zx_dim != 0:
            pzx = dist.Normal(zx_p_loc, zx_p_scale)
        else:
            pzx = None
        pzy = dist.Normal(zy_p_loc, zy_p_scale)

        d_hat = self.qd(zd_q)
        y_hat = self.qy(zy_q)
        return x_recon, d_hat, y_hat, qzd, pzd, zd_q, qzx, pzx, zx_q, qzy, pzy, zy_q
    
    # Pérdida
    def loss_function(self, d, x, y=None, c=None): 
        device = x.device
        
        if y is None:  # unsupervised
            # --- INFERENCIA ---
            zd_q_loc, zd_q_scale = self.qzd(x); qzd = dist.Normal(zd_q_loc, zd_q_scale); zd_q = qzd.rsample()
            if self.zx_dim != 0: zx_q_loc, zx_q_scale = self.qzx(x); qzx = dist.Normal(zx_q_loc, zx_q_scale); zx_q = qzx.rsample()
            else: zx_q = None
            zy_q_loc, zy_q_scale = self.qzy(x); qzy = dist.Normal(zy_q_loc, zy_q_scale); zy_q = qzy.rsample()
            
            # --- GENERATIVE PRIORS ---
            zd_p_loc, zd_p_scale = self.pzd(d)
            if self.zx_dim != 0:
                zx_p_loc, zx_p_scale = torch.zeros(zd_p_loc.size()[0], self.zx_dim, device=device), \
                    torch.ones(zd_p_loc.size()[0], self.zx_dim, device=device)
            pzd = dist.Normal(zd_p_loc, zd_p_scale)
            if self.zx_dim != 0: pzx = dist.Normal(zx_p_loc, zx_p_scale)
            else: pzx = None
            d_hat = self.qd(zd_q)
            
            # Reconstruction - PASAR C AL DECODER
            x_recon = self.px(zd_q, zx_q, zy_q, c)
            
            # 🚨 CORRECCIÓN CLAVE: RECONSTRUCCIÓN CON L2/MSE
            x_recon_final = x_recon.mean(dim=1, keepdim=True) # (N, 1, 8, 52)
            x_target = x # (N, 1, 8, 52)
            L2_x = F.mse_loss(x_recon_final, x_target, reduction='sum')
            
            # KL Divergence (zd)
            zd_p_minus_zd_q = torch.sum(pzd.log_prob(zd_q) - qzd.log_prob(zd_q))
            # ... (KL_zx)
            
            # Classification loss of subject
            _, d_target = d.max(dim=1)
            CE_d = F.cross_entropy(d_hat, d_target.long(), reduction='sum') # Cast a Long
            
            # --- Marginalización sobre y ---
            batch_size = x.size(0)
            y_indices = torch.arange(self.y_dim, device=device).repeat(batch_size) 
            y_onehot = torch.eye(self.y_dim, device=device)[y_indices]
            
            # Repetir zy_q y sus parámetros para cada y_onehot (self.y_dim veces)
            zy_q = zy_q.repeat_interleave(self.y_dim, dim=0)
            zy_q_loc, zy_q_scale = zy_q_loc.repeat_interleave(self.y_dim, dim=0), zy_q_scale.repeat_interleave(self.y_dim, dim=0)
            qzy = dist.Normal(zy_q_loc, zy_q_scale)
            
            zy_p_loc, zy_p_scale = self.pzy(y_onehot); pzy = dist.Normal(zy_p_loc, zy_p_scale)
            y_hat = self.qy(zy_q)
            
            # Marginals
            alpha_y = F.softmax(y_hat, dim=-1); qy = dist.OneHotCategorical(alpha_y)
            prob_qy = torch.exp(qy.log_prob(y_onehot))
            
            # KL Divergence
            zy_p_minus_zy_q = torch.sum(pzy.log_prob(zy_q) - qzy.log_prob(zy_q), dim=-1)
            marginal_zy_p_minus_zy_q = torch.sum(prob_qy * zy_p_minus_zy_q)
            prior_y = torch.tensor(1 / self.y_dim, device=device)
            prior_y_minus_qy = torch.log(prior_y) - qy.log_prob(y_onehot)
            marginal_prior_y_minus_qy = torch.sum(prob_qy * prior_y_minus_qy)
            
            # Total loss
            return L2_x \
                - self.beta_d * zd_p_minus_zd_q \
                - self.beta_x * KL_zx \
                - self.beta_y * marginal_zy_p_minus_zy_q \
                - marginal_prior_y_minus_qy \
                + self.aux_loss_multiplier_d * CE_d

        else:  # supervised
            x_recon, d_hat, y_hat, qzd, pzd, zd_q, qzx, pzx, zx_q, qzy, pzy, zy_q = self.forward(d, x, y, c) 
            
            # 🚨 CORRECCIÓN CLAVE: RECONSTRUCCIÓN CON L2/MSE
            x_recon_final = x_recon.mean(dim=1, keepdim=True) # (N, 1, 8, 52)
            x_target = x # (N, 1, 8, 52)
            L2_x = F.mse_loss(x_recon_final, x_target, reduction='sum')
            
            # KL Divergence (sin cambios)
            zd_p_minus_zd_q = torch.sum(pzd.log_prob(zd_q) - qzd.log_prob(zd_q))
            if self.zx_dim != 0: KL_zx = torch.sum(pzx.log_prob(zx_q) - qzx.log_prob(zx_q))
            else: KL_zx = 0
            zy_p_minus_zy_q = torch.sum(pzy.log_prob(zy_q) - qzy.log_prob(zy_q))
            
            # classification loss of subject and gesture
            _, d_target = d.max(dim=1)
            CE_d = F.cross_entropy(d_hat, d_target.long(), reduction='sum') # 🚨 CORRECCIÓN: Cast a Long
            
            _, y_target = y.max(dim=1)
            CE_y = F.cross_entropy(y_hat, y_target.long(), reduction='sum') # 🚨 CORRECCIÓN: Cast a Long
            
            # Total loss
            return L2_x \
                   - self.beta_d * zd_p_minus_zd_q \
                   - self.beta_x * KL_zx \
                   - self.beta_y * zy_p_minus_zy_q \
                   + self.aux_loss_multiplier_d * CE_d \
                   + self.aux_loss_multiplier_y * CE_y, \
                CE_y, zd_q, zy_q, zx_q, d_target, y_target

    # Clasificador
    def classifier(self, x):
        """
        clasificar una imagen (o un lote de imágenes)
        :param xs: un lote de vectores escalados de píxeles de una imagen
        :return: un lote de las correspondientes etiquetas de clase (como one-hots)
        """
        with torch.no_grad():
            zd_q_loc, zd_q_scale = self.qzd(x)
            zd = zd_q_loc
            alpha = F.softmax(self.qd(zd), dim=1)
            # obtiener el índice (dígito) que corresponde a la máxima probabilidad de clase predicha
            res, ind = torch.topk(alpha, 1)
            # Convirtir el(los) dígito(s) a tensor(es) one-hot
            d = x.new_zeros(alpha.size())
            d = d.scatter_(1, ind, 1.0)
            # Obtener la etiqueta de gesto
            zy_q_loc, zy_q_scale = self.qzy.forward(x)
            zy = zy_q_loc
            alpha = F.softmax(self.qy(zy), dim=1)
            # obtener el índice (dígito) que corresponde a la máxima probabilidad de clase predicha
            res, ind = torch.topk(alpha, 1)
            # convertir el(los) dígito(s) a tensor(es) one-hot
            y = x.new_zeros(alpha.size())
            y = y.scatter_(1, ind, 1.0)
        # retornar las etiquetas de clase
        return d, y

    # Clasificador para t-SNE
    def classifier_for_tsne(self, x):
        """
        clasificar una imagen (o un lote de imágenes)
        :param xs: un lote de vectores escalados de píxeles de una imagen
        :return: un lote de las correspondientes etiquetas de clase (como one-hots)
        """
        with torch.no_grad():
            zd_q_loc, zd_q_scale = self.qzd(x)
            zd = zd_q_loc
            alpha = F.softmax(self.qd(zd), dim=1)
            # obtener el índice (dígito) que corresponde a la máxima probabilidad de clase predicha
            res, ind = torch.topk(alpha, 1) 
            # convertir el(los) dígito(s) a tensor(es) one-hot
            d = x.new_zeros(alpha.size())
            d = d.scatter_(1, ind, 1.0)
            # Obtener la etiqueta de gesto
            zy_q_loc, zy_q_scale = self.qzy.forward(x)
            zy = zy_q_loc
            alpha = F.softmax(self.qy(zy), dim=1)
            # obtener el índice (dígito) que corresponde a la máxima probabilidad de clase predicha
            res, ind = torch.topk(alpha, 1)
            # convertir el(los) dígito(s) a tensor(es) one-hot
            y = x.new_zeros(alpha.size())
            y = y.scatter_(1, ind, 1.0)
        # retornar las etiquetas de clase
        return d, y, zy
